from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from banabert_cls import LitBanaBERForSeqClassifier, train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_banabert_cls")

    # model
    model_ck = "banabert_model/ot_cl"
    num_classes = 5
    lit_banabert_cls = LitBanaBERForSeqClassifier(num_classes, model_ck)

    # train model
    trainer = pl.Trainer(
        max_epochs=20, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp",
        callbacks=[EarlyStopping(monitor="valid/acc_epoch", min_delta=0.00, patience=5, verbose=False, mode="max")]
    )
    trainer.fit(model=lit_banabert_cls, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=lit_banabert_cls, dataloaders=test_dataloader)
