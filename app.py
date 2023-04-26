from potassium import Potassium, Request, Response
from faster_whisper import WhisperModel

model_size = "tiny"

app = Potassium("potassium_app_faster_whisper")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    context = {
        "model": model
    }
    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    segments, info = model.transcribe("first_30_seconds.mp3", beam_size=5, vad_filter=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    segments_list = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        segments_list.append(segment.text)

    return Response(
        json={"transcription": segments_list},
        status=200
    )


if __name__ == "__main__":
    app.serve()
