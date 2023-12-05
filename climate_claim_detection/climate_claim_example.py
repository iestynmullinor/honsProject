from transformers import pipeline


# this mode is actually very good
CLIMATE_CLAIM_DETECTOR_MODEL = "climatebert/distilroberta-base-climate-detector"
claim_detector = pipeline("text-classification", model=CLIMATE_CLAIM_DETECTOR_MODEL)

if __name__=="__main__":
    while True:
        input_sentence = input("Enter sentence: ")

        claim_detector_output = claim_detector([input_sentence])
        if claim_detector_output[0]['label'] == 'no':
            print("This sentence is not a climate claim, so no evidence will be retrieved.")
        else:
            print("This sentence is a climate claim, so evidence will be retrieved.")
        print("------------------------------------------------------------------------------------------")