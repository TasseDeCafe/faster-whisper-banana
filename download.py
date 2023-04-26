# This file runs during container build time to get model weights built into the container

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    from faster_whisper import WhisperModel

    model_size = "tiny"

    # Run on GPU with FP16
    WhisperModel(model_size, device="cpu", compute_type="int8", download_root="models")

if __name__ == "__main__":
    download_model()