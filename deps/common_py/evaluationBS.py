"""Binary-segmentation metrics used by the LTS demo and experiments.

Ground-truth and prediction masks use the CDNet convention: foreground is
``255`` and background is ``0``.  Pixels with other values are ignored, just
as they were in the original implementation.
"""

import numpy as np
import torch


def normalize_mask(mask):
    """Return an image-shaped mask with a single label channel.

    CDNet ground-truth files are often stored as RGB PNGs even though all
    channels carry the same label.  The released LTS notebook mixes those
    RGB masks with single-channel predictions, so normalize the common
    ``(height, width, channels)`` representation at the evaluation boundary.
    Two-dimensional masks and tensors with a different layout are returned
    unchanged.
    """

    if mask.ndim == 3 and mask.shape[-1] in (1, 3, 4):
        return mask[..., 0]

    return mask


def evaluation_numpy_entry_torch(prelabs, trulabs):
    """Return ``(TP, FP, TN, FN)`` as NumPy scalar arrays for tensors."""

    prelabs = normalize_mask(prelabs)
    trulabs = normalize_mask(trulabs)

    tp = torch.sum((prelabs == 255) & (trulabs == 255))
    fp = torch.sum((prelabs == 255) & (trulabs == 0))
    tn = torch.sum((prelabs == 0) & (trulabs == 0))
    fn = torch.sum((prelabs == 0) & (trulabs == 255))

    # ``detach().cpu()`` preserves the returned NumPy scalar shape while also
    # allowing callers to evaluate tensors produced on a CUDA device.
    return (
        tp.detach().cpu().numpy(),
        fp.detach().cpu().numpy(),
        tn.detach().cpu().numpy(),
        fn.detach().cpu().numpy(),
    )


def _confusion_counts(prelabs, trulabs):
    """Count the four binary-mask outcomes without changing label semantics."""

    prelabs = normalize_mask(prelabs)
    trulabs = normalize_mask(trulabs)

    # Flatten to scalars before conversion.  The previous (n, 1) indexing
    # raised under NumPy 2.x because ``int(np.array([value]))`` is no longer
    # accepted; the classification rules themselves remain unchanged.
    predictions = np.reshape(prelabs, -1)
    truth = np.reshape(trulabs, -1)

    tp = tn = fp = fn = 0

    # Use the prediction length as the loop bound, matching the old function
    # for mismatched input lengths (an undersized truth array still raises).
    for i in range(np.size(predictions)):
        prolab = int(predictions[i])
        trulab = int(truth[i])

        if prolab == 255 and trulab == 255:
            tp += 1

        if prolab == 255 and trulab == 0:
            fp += 1

        if prolab == 0 and trulab == 0:
            tn += 1

        if prolab == 0 and trulab == 255:
            fn += 1

    return tp, fp, tn, fn


def evaluation_numpy_entry(prelabs, trulabs):
    """Return integer ``(TP, FP, TN, FN)`` counts for NumPy-like masks."""

    return _confusion_counts(prelabs, trulabs)


def _metrics(tp, fp, fn):
    """Compute recall, precision and F-measure using the original safeguards."""

    recall = tp / max((tp + fn), 1)
    precision = tp / max((tp + fp), 1)
    f_measure = (2 * precision * recall) / max((precision + recall), 0.0001)

    return recall, precision, f_measure


def evaluation_numpy(prelabs, trulabs):
    """Return ``(recall, precision, F-measure)`` for binary masks."""

    tp, fp, _tn, fn = _confusion_counts(prelabs, trulabs)
    return _metrics(tp, fp, fn)


def evaluation_BS(prelabs, trulabs):
    """Return metrics followed by the underlying ``TP, FP, TN, FN`` counts."""

    tp, fp, tn, fn = _confusion_counts(prelabs, trulabs)
    recall, precision, f_measure = _metrics(tp, fp, fn)

    return recall, precision, f_measure, tp, fp, tn, fn
