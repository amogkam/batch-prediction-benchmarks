import click
import os
import time
import ray
from ray.data.datasource import ImageFolderDatasource
from ray.data.preprocessors import BatchMapper
from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchCheckpoint, TorchPredictor

import pandas as pd
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from theoretical import BATCH_SIZE

CACHED_DATASET_PATH = "/tmp/cached_ds.parquet"

@click.command("Run GPU batch prediction for image classification.")
@click.option("--separate-stages", type=bool, default=True, help="If set, calculate times for each stage separately. Avoids all Dataset lazy execution and stage fusion. Useful for benchmarking stages independently.")
@click.option("--predict-only", type=bool, default=False, help="If set, use cached preprocessed dataset. Useful for benchmarking only prediction.")
def main(separate_stages, predict_only):
    stage_times = {"read": 0.0, "preprocess": 0.0, "predict": 0.0}
    start_time = time.time()
    if predict_only:
        if not os.path.exists(CACHED_DATASET_PATH):
            raise ValueError("Attempting to use cached preprocessed dataset, but it does not exist. To create the cached dataset, first run benchmarking with the --separate-stages flag.")
        ds = ray.data.read_parquet(CACHED_DATASET_PATH)
        preprocessor = None
    else:
        read_time_start = time.time()
        ds = ray.data.read_datasource(
            ImageFolderDatasource(), root="/home/ray/batch-prediction-benchmark/imagenet/data")

        read_time_end = time.time()
        if separate_stages:
            ds.fully_executed()
            read_time_end = time.time()
        
        stage_times["read"] = read_time_end - read_time_start

        def preprocess(df: pd.DataFrame) -> pd.DataFrame:
            """
            User Pytorch code to transform user image.

            Taken from https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/python/dataset.py#L205-L218
            """
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
                ]
            )
            processed_image_df = pd.DataFrame({"image": [preprocess(image).numpy() for image in df["image"]]})
            return processed_image_df

        preprocess_time_start = time.time()
        preprocessor = BatchMapper(preprocess)

        preprocess_time_end = time.time()
        if separate_stages:
            ds = preprocessor.transform(ds)
            ds.fully_executed()
            preprocess_time_end = time.time()
            if not os.path.exists(CACHED_DATASET_PATH):
                ds.write_parquet(CACHED_DATASET_PATH)
            preprocessor = None

        stage_times["preprocess"] = preprocess_time_end - preprocess_time_start

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    ckpt = TorchCheckpoint.from_model(model=model, preprocessor=preprocessor)
    prediction_time_start = time.time()
    predictor = BatchPredictor.from_checkpoint(ckpt, TorchPredictor)
    predictor.predict(ds, num_gpus_per_worker=1, feature_columns=["image"], batch_size=BATCH_SIZE)
    prediction_time_end = time.time()
    stage_times["predict"] = prediction_time_end - prediction_time_start

    if separate_stages:
        print("Times for each stage: ", stage_times)
    
    overall_time = sum(stage_times.values())
    print("Total time: ", overall_time)
    print(f"Throughput {ds.count() / overall_time} (img/sec)")

if __name__ == "__main__":
    main()







