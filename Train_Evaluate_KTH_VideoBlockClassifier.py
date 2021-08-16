from KTH_DataModule import KTH_DataModule
from KTH_VideoBlockClassifier import KTH_VideoBlockClassifier
import pytorch_lightning as pl

if __name__ == "__main__":
    data_module = KTH_DataModule('C:/Users/s441606/Documents/Videos/KTHA')

    # train
    model = KTH_VideoBlockClassifier()
    trainer = pl.Trainer()
    trainer.fit(model, data_module)

    # evaluate
    trainer.test(model,data_module)