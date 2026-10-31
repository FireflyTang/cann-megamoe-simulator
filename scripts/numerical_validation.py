"""Explicit output-precision policies shared by runtime and offline acceptance.

Kernel/routing/count/shape contracts stay separate. No device access or hidden
output copy is performed here: runtime passes its existing CPU comparison data.
"""
import math
from measurement_validation import validate_tolerances

STRICT = 'strict_elementwise_v1'
MIXED = 'bf16_mixed_ratio_v1'
MIXED_L2 = 'bf16_mixed_ratio_l2_v2'
L2_METRIC = 'scaled_l2_error_over_reference_v1'


class NumericalValidationError(AssertionError):
    """A failed output comparison, retaining its true JSON-safe statistics."""
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('Numerical output rejected: ' + ', '.join(evidence['reasons']) +
                         '; mismatches=' + str(evidence.get('mismatches')) +
                         '; matched_ratio=' + str(evidence.get('matched_ratio')) +
                         '; max_abs_error=' + str(evidence.get('max_abs_error')))


def _validation(value):
    if not isinstance(value, dict):
        raise ValueError('validation must be an object')
    return value['validation'] if 'validation' in value else value


def _number(value, name, minimum=0, maximum=None):
    if type(value) not in (int, float):
        raise ValueError(name + ' must be finite numeric (not bool/string)')
    try:
        valid = math.isfinite(value) and value >= minimum and (maximum is None or value <= maximum)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError(name + ' is outside its finite allowed range')
    return float(value)


def validate_policy(validation):
    """Normalize an explicit mixed policy, otherwise retain strict 100% matching."""
    value = _validation(validation)
    validate_tolerances(value)
    name = value.get('numerical_policy', STRICT)
    if name != MIXED_L2 and 'max_relative_l2_error' in value:
        raise ValueError('max_relative_l2_error requires bf16_mixed_ratio_l2_v2')
    if name not in (STRICT, MIXED, MIXED_L2):
        raise ValueError('Unsupported validation.numerical_policy: ' + repr(name))
    if name == MIXED_L2:
        if 'max_abs_error_limit' in value:
            raise ValueError('L2 v2 cannot also enable max_abs_error_limit')
        ratio = _number(value.get('required_matched_ratio', .99), 'required_matched_ratio', minimum=.99, maximum=1)
        limit = _number(value.get('max_relative_l2_error', 1/64), 'max_relative_l2_error', maximum=1/64)
        if value['rtol'] > 1/64 or value['atol'] > 1/64:
            raise ValueError('bf16_mixed_ratio_l2_v2 cannot relax rtol/atol beyond 1/64')
        return dict(numerical_policy=name, policy=name, rtol=float(value['rtol']),
            atol=float(value['atol']), required_matched_ratio=ratio, max_relative_l2_error=limit)
    if name == MIXED:
        ratio = _number(value.get('required_matched_ratio', .99), 'required_matched_ratio', minimum=.99, maximum=1)
        cap = _number(value.get('max_abs_error_limit', 1), 'max_abs_error_limit', maximum=1)
        if value['rtol'] > 1/64 or value['atol'] > 1/64:
            raise ValueError('bf16_mixed_ratio_v1 cannot relax rtol/atol beyond 1/64')
    else:
        ratio = _number(value.get('required_matched_ratio', 1), 'required_matched_ratio', maximum=1)
        if ratio != 1 or value.get('max_abs_error_limit') is not None:
            raise ValueError('Ratio/cap relaxation requires an explicit bf16_mixed_ratio_v1 policy')
        cap = None
    return dict(numerical_policy=name, policy=name, rtol=float(value['rtol']),
                atol=float(value['atol']), required_matched_ratio=ratio,
                max_abs_error_limit=cap)


def _relative_l2(error_scale, error_ssq, reference_scale, reference_ssq,
                 reference_zero, error_zero):
    """Reconstruct a dimensionless norm without overflowing either raw norm."""
    if reference_zero:
        return (0.0, True, False) if error_zero else (None, False, False)
    if error_zero:
        return 0.0, True, False
    if error_ssq == 0:
        return None, False, False
    em, ee = math.frexp(error_scale)
    rm, re = math.frexp(reference_scale)
    factor = math.sqrt(error_ssq) / math.sqrt(reference_ssq) * (em / rm)
    try:
        result = math.ldexp(factor, ee - re)
    except OverflowError:
        return None, False, True
    if not math.isfinite(result):
        return None, False, True
    if result == 0 and not error_zero:
        return None, False, False
    return result, True, False


def _l2_evidence(actual, expected):
    import torch
    a, b = actual.double(), expected.double()
    reference_scale = float(b.abs().max())
    difference = a - b
    maximum = float(difference.abs().max())
    error_scale = maximum if math.isfinite(maximum) else max(float(a.abs().max()), reference_scale)
    reference_zero = reference_scale == 0
    error_zero = bool((actual == expected).all())
    if reference_zero:
        reference_ssq = 0.0
    else:
        reference_ssq = float((b / reference_scale).square().sum())
    if error_scale == 0 or error_zero:
        error_ssq = 0.0
    else:
        # When a finite FP64 subtraction itself overflows, subtract scaled
        # operands. Ordinary finite differences keep their original accuracy.
        scaled = torch.where(torch.isfinite(difference), difference / error_scale,
                             a / error_scale - b / error_scale)
        error_ssq = float(scaled.square().sum())
    ratio, finite, overflow = _relative_l2(error_scale, error_ssq, reference_scale,
        reference_ssq, reference_zero, error_zero)
    return dict(l2_metric=L2_METRIC, l2_error_scale=error_scale,
        l2_error_scaled_sum_squares=error_ssq, l2_reference_scale=reference_scale,
        l2_reference_scaled_sum_squares=reference_ssq, l2_reference_zero=reference_zero,
        l2_error_zero=error_zero, relative_l2=ratio, relative_l2_finite=finite,
        relative_l2_overflow=overflow)


def _validate_l2_evidence(rule, evidence, total, maximum):
    if 'max_abs_error_limit' in evidence or evidence.get('l2_metric') != L2_METRIC:
        raise ValueError('Incorrect L2 v2 evidence definition or mixed absolute cap')
    error_scale = _number(evidence.get('l2_error_scale'), 'l2_error_scale')
    reference_scale = _number(evidence.get('l2_reference_scale'), 'l2_reference_scale')
    error_ssq = _number(evidence.get('l2_error_scaled_sum_squares'), 'l2_error_scaled_sum_squares', maximum=4*total)
    reference_ssq = _number(evidence.get('l2_reference_scaled_sum_squares'), 'l2_reference_scaled_sum_squares', maximum=total)
    reference_zero = evidence.get('l2_reference_zero')
    error_zero = evidence.get('l2_error_zero')
    if type(reference_zero) is not bool or type(error_zero) is not bool:
        raise ValueError('L2 zero-norm markers must be boolean')
    if type(evidence.get('expected_all_zero')) is not bool or type(evidence.get('actual_all_zero')) is not bool:
        raise ValueError('L2 requires explicit actual/reference zero markers')
    if (reference_zero is not evidence['expected_all_zero'] or reference_zero != (reference_scale == 0)            or reference_zero != (reference_ssq == 0) or error_zero != (maximum == 0)            or error_zero != (error_ssq == 0) or error_zero != (error_scale == 0)):
        raise ValueError('Inconsistent L2 zero-norm/scaled-norm statistics')
    if not reference_zero and reference_ssq < 1:
        raise ValueError('Nonzero scaled reference norm must contain its maximum element')
    if error_zero and evidence['mismatches'] != 0:
        raise ValueError('A zero L2 numerator disagrees with the comparison statistics')
    if reference_zero and evidence['actual_all_zero'] is not error_zero:
        raise ValueError('Zero reference with nonzero output cannot have zero relative error')
    if not reference_zero and evidence['actual_all_zero'] and error_zero:
        raise ValueError('Zero actual and nonzero reference cannot have zero error')
    if error_scale != maximum:
        raise ValueError('Scaled L2 numerator scale disagrees with maximum absolute error')
    if maximum > 0:
        if error_scale == 0:
            raise ValueError('Nonzero absolute error has zero norm scale')
        scaled_maximum_ssq = (maximum / error_scale) ** 2
        if (error_ssq < scaled_maximum_ssq * (1-1e-12)                or error_ssq > total * scaled_maximum_ssq * (1+1e-12)):
            raise ValueError('Scaled error norm contradicts maximum absolute error')
    ratio, finite, overflow = _relative_l2(error_scale, error_ssq, reference_scale,
        reference_ssq, reference_zero, error_zero)
    if (evidence.get('relative_l2_finite') is not finite            or evidence.get('relative_l2_overflow') is not overflow or not finite):
        raise ValueError('Invalid/nonfinite relative L2 evidence')
    reported = _number(evidence.get('relative_l2'), 'relative_l2')
    if not math.isclose(reported, ratio, rel_tol=1e-12, abs_tol=0):
        raise ValueError('Relative L2 disagrees with scaled numerator/denominator')
    if reported > rule['max_relative_l2_error']:
        raise ValueError('Relative L2 exceeds the configured limit')


def check_output(actual, expected, validation):
    """Compare CPU tensors once; fail with statistics instead of hiding outliers."""
    import torch
    rule = validate_policy(validation)
    evidence = dict(rule, status='fail', classification='numerical_error', reasons=[],
        finite=False, elements=0, mismatches=None, matched_elements=None, matched_ratio=None,
        max_abs_error=None, strict_all_elements_pass=False, dtype=None, shape=None,
        actual_dtype=None, expected_dtype=None, actual_finite_elements=0, expected_finite_elements=0)
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        evidence['reasons'].append('tensor_type_mismatch')
        raise NumericalValidationError(evidence)
    evidence.update(dtype=str(actual.dtype), actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
        shape=list(actual.shape), expected_shape=list(expected.shape), elements=actual.numel())
    if actual.device.type != 'cpu' or expected.device.type != 'cpu':
        evidence['reasons'].append('comparison_requires_existing_cpu_tensors')
    if actual.shape != expected.shape:
        evidence['reasons'].append('shape_mismatch')
    if (actual.dtype != expected.dtype or actual.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.float64)):
        evidence['reasons'].append('dtype_mismatch')
    if actual.numel() == 0 or actual.ndim == 0:
        evidence['reasons'].append('empty_or_scalar_output')
    if evidence['reasons']:
        raise NumericalValidationError(evidence)
    actual_finite = int(torch.isfinite(actual).sum())
    expected_finite = int(torch.isfinite(expected).sum())
    finite = actual_finite == expected_finite == actual.numel()
    evidence.update(finite=finite, actual_finite_elements=actual_finite,
        expected_finite_elements=expected_finite, actual_all_zero=bool((actual == 0).all()),
        expected_all_zero=bool((expected == 0).all()))
    if not finite:
        evidence['reasons'].append('nonfinite_actual_or_expected')
        raise NumericalValidationError(evidence)
    close = torch.isclose(actual, expected, rtol=rule['rtol'], atol=rule['atol'])
    matched = int(close.sum())
    elements = actual.numel()
    # FP64 here only prevents overflow in the reported absolute difference; it
    # does not change the reference, comparison tolerance or kernel arithmetic.
    maximum = float((actual.double() - expected.double()).abs().max())
    evidence.update(matched_elements=matched, mismatches=elements-matched, matched_ratio=matched/elements,
        max_abs_error=maximum, strict_all_elements_pass=matched == elements)
    if rule['numerical_policy'] in (MIXED, MIXED_L2):
        significant_zero = ((actual == 0) & ~close).reshape(-1, actual.shape[-1])
        row_counts = significant_zero.sum(dim=1)
        longest = 0
        # Highest counts first: candidate rows with >=16 points are checked first;
        # once their count cannot improve the maximum no row scan is needed.
        for row in row_counts.argsort(descending=True).tolist():
            if int(row_counts[row]) <= longest:
                break
            columns = significant_zero[row].nonzero().flatten().tolist()
            current = 0
            previous = -2
            for column in columns:
                current = current + 1 if column == previous + 1 else 1
                longest = max(longest, current)
                previous = column
        evidence.update(zero_dropout_rule='significant_zero_run16_v1',
            zero_dropout_count=int(row_counts.sum()), zero_dropout_max_run=longest,
            zero_dropout_fail=longest >= 16)
        if longest >= 16:
            evidence['reasons'].append('structural_zero_dropout')
    if matched/elements < rule['required_matched_ratio']:
        evidence['reasons'].append('matched_ratio_below_required')
    if rule.get('max_abs_error_limit') is not None and maximum > rule['max_abs_error_limit']:
        evidence['reasons'].append('max_abs_error_limit_exceeded')
    if rule['numerical_policy'] == MIXED_L2:
        evidence.update(_l2_evidence(actual, expected))
        if not math.isfinite(maximum):
            evidence['max_abs_error'] = None
            evidence['reasons'].append('absolute_error_stat_overflow')
        if not evidence['relative_l2_finite']:
            evidence['reasons'].append('zero_reference_nonzero_output' if evidence['l2_reference_zero']
                                       else 'relative_l2_nonfinite_or_underflow')
        elif evidence['relative_l2'] > rule['max_relative_l2_error']:
            evidence['reasons'].append('relative_l2_limit_exceeded')
    if evidence['reasons']:
        raise NumericalValidationError(evidence)
    evidence.update(status='pass', classification=('strict_all_elements_pass' if matched == elements else
                                                  'within_aggregate_tolerance'))
    return evidence


def validate_precision_evidence(validation, evidence, expected_elements=None):
    """Validate saved true statistics and reapply the frozen acceptance policy.

    This cannot replace an output comparison; it checks its identity, arithmetic
    consistency and acceptance decision. Legacy packages without these fields
    remain the caller's unchanged legacy-validation branch.
    """
    rule = validate_policy(validation)
    if not isinstance(evidence, dict):
        raise ValueError('Numerical evidence must be an object')
    for field in rule:
        if field not in evidence or evidence[field] != rule[field] or (
                isinstance(evidence[field], bool) and type(rule[field]) is not bool):
            raise ValueError('Numerical evidence/config mismatch: ' + field)
    if evidence.get('status') != 'pass' or evidence.get('finite') is not True or evidence.get('reasons') != []:
        raise ValueError('Numerical evidence reports failed/nonfinite output')
    total = evidence.get('elements')
    mismatches = evidence.get('mismatches')
    matched = evidence.get('matched_elements')
    if type(total) is not int or total <= 0:
        raise ValueError('Invalid numerical element count')
    if expected_elements is not None and (type(expected_elements) is not int or total != expected_elements):
        raise ValueError('Numerical evidence element count differs from expected shape')
    if type(mismatches) is not int or not 0 <= mismatches <= total:
        raise ValueError('Invalid numerical mismatch count')
    if type(matched) is not int or matched != total - mismatches:
        raise ValueError('Matched/mismatched numerical counts disagree')
    ratio = _number(evidence.get('matched_ratio'), 'evidence.matched_ratio', maximum=1)
    if not math.isclose(ratio, matched / total, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Numerical matched ratio disagrees with true counts')
    maximum = _number(evidence.get('max_abs_error'), 'evidence.max_abs_error')
    if evidence.get('strict_all_elements_pass') is not (mismatches == 0):
        raise ValueError('Numerical strict comparison flag disagrees with mismatches')
    shape = evidence.get('shape')
    if (not isinstance(shape, list) or not shape or any(type(v) is not int or v <= 0 for v in shape)
            or math.prod(shape) != total):
        raise ValueError('Numerical shape/element count mismatch')
    if evidence.get('dtype') not in ('torch.bfloat16', 'torch.float32', 'torch.float16', 'torch.float64'):
        raise ValueError('Unsupported numerical output dtype')
    if (evidence.get('actual_dtype') != evidence.get('expected_dtype')
            or evidence.get('actual_dtype') not in ('torch.bfloat16', 'torch.float32', 'torch.float16', 'torch.float64')):
        raise ValueError('Numerical comparison dtype mismatch')
    if evidence.get('expected_shape') != shape:
        raise ValueError('Numerical comparison shape mismatch')
    if (evidence.get('actual_finite_elements') != total or evidence.get('expected_finite_elements') != total
            or type(evidence.get('actual_finite_elements')) is not int
            or type(evidence.get('expected_finite_elements')) is not int):
        raise ValueError('Numerical finite counts disagree')
    if ratio < rule['required_matched_ratio'] or (rule.get('max_abs_error_limit') is not None and maximum > rule['max_abs_error_limit']):
        raise ValueError('Numerical evidence violates the configured ratio or absolute cap')
    if rule['numerical_policy'] in (MIXED, MIXED_L2):
        count = evidence.get('zero_dropout_count')
        longest = evidence.get('zero_dropout_max_run')
        if (evidence.get('zero_dropout_rule') != 'significant_zero_run16_v1'
                or type(count) is not int or not 0 <= count <= mismatches
                or type(longest) is not int or not 0 <= longest <= min(count, shape[-1])
                or (count == 0) != (longest == 0)
                or evidence.get('zero_dropout_fail') is not (longest >= 16)
                or longest >= 16):
            raise ValueError('Structural zero-dropout evidence is invalid or failed')
    if rule['numerical_policy'] == MIXED_L2:
        _validate_l2_evidence(rule, evidence, total, maximum)
    classification = 'strict_all_elements_pass' if mismatches == 0 else 'within_aggregate_tolerance'
    if evidence.get('classification') != classification:
        raise ValueError('Numerical classification disagrees with comparison')
    return dict(evidence)
