import load
def predict():
    print("PREDICTION")
def train():
    print("TRAINING")
def validate():
    print("VALIDATION")

def main():
    print("1. Train the model")
    print("2. Predict the digit")
    print("3. Validate the model")
    c = int(input())
    if c == 1:
        train()
    elif c == 2:
        predict()
    else:
        validate()

if __name__ =='__main__':
    main()
