from DialogueQualityEvaluator.dialoguequalityevaluator import *

from responder import Responder

model_path = "./seq2seq.pt"
spm_path = "./sentencepiece.model"
model = Responder.load(model_path, spm_path)


# input: List of input sentences. type: list(str)
# output: List of candidate reply messages. type: list(str)
def pseudo_model(input_logs):
    res = model.predict_sentences(input_logs, max_output_len=10, device='cuda:0', noise_gain=0.2)
    return res

def main():
    # Run test.
    score = evaluate_dialogue_quality_of_model("./DialogueQualityEvaluator/test.txt", pseudo_model, skip_probability=0.99, max_reference_logs=6, min_reference_logs=6)
    print(f"Score: {score}")

if __name__ == '__main__':
    main()
