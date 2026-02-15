import unittest

try:
    import torch

    from engine import _compute_discriminator_loss
    from losses import AdversarialLoss
    _DEPS_AVAILABLE = True
except ModuleNotFoundError:
    _DEPS_AVAILABLE = False


@unittest.skipUnless(_DEPS_AVAILABLE, "torch dependency is required")
class GanLossParityTest(unittest.TestCase):
    def test_discriminator_loss_parity_with_same_labels(self):
        criterion = AdversarialLoss()
        d_real = torch.tensor([[0.25], [-1.0], [1.5]], dtype=torch.float32)
        d_fake = torch.tensor([[-0.3], [0.6], [-2.0]], dtype=torch.float32)

        loss_train, loss_train_real, loss_train_fake = _compute_discriminator_loss(
            d_real,
            d_fake,
            criterion,
            real_label=0.9,
            fake_label=0.0,
        )
        loss_val, loss_val_real, loss_val_fake = _compute_discriminator_loss(
            d_real,
            d_fake,
            criterion,
            real_label=0.9,
            fake_label=0.0,
        )

        self.assertAlmostEqual(loss_train.item(), loss_val.item(), places=7)
        self.assertAlmostEqual(loss_train_real.item(), loss_val_real.item(), places=7)
        self.assertAlmostEqual(loss_train_fake.item(), loss_val_fake.item(), places=7)

    def test_discriminator_loss_changes_with_label_policy(self):
        criterion = AdversarialLoss()
        d_real = torch.tensor([[0.4], [0.8]], dtype=torch.float32)
        d_fake = torch.tensor([[-0.4], [0.2]], dtype=torch.float32)

        loss_soft, _, _ = _compute_discriminator_loss(
            d_real,
            d_fake,
            criterion,
            real_label=0.9,
            fake_label=0.0,
        )
        loss_hard, _, _ = _compute_discriminator_loss(
            d_real,
            d_fake,
            criterion,
            real_label=1.0,
            fake_label=0.0,
        )

        self.assertNotAlmostEqual(loss_soft.item(), loss_hard.item(), places=6)


if __name__ == "__main__":
    unittest.main()
