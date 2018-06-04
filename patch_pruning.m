function [Xh, Xl] = patch_pruning(Xh, Xl, threshold)

pvars = var(Xh, 0);

idx = pvars > threshold;

Xh = Xh(:, idx);
Xl = Xl(:, idx);