import os

def run_data_prep():
    from data_prep import main as data_prep_main
    print("\n=== Running Data Preparation ===")
    data_prep_main()

def run_gan_training():
    from train_gan import train
    print("\n=== Running GAN Training ===")
    train()

def run_generate_fakes():
    from generate_fakes import generate_fakes
    print("\n=== Generating Fake Images ===")
    generate_fakes(num_samples=500)

def run_cnn_training():
    from train_signature_classifier import train_classifier
    print("\n=== Training CNN Classifier ===")
    train_classifier()

def run_gan_evaluation():
    from evaluate_gan_with_cnn import evaluate_gan_images
    print("\n=== Evaluating GAN Images with CNN ===")
    cnn_model_path = "best_signature_cnn.pth"
    gan_images_folder = os.path.join("data", "generated_fakes")
    evaluate_gan_images(cnn_model_path, gan_images_folder)

if __name__ == "__main__":
    # Uncomment/comment steps as needed
    #run_data_prep()
    #run_gan_training()
    #run_generate_fakes()
    #run_cnn_training()
    run_gan_evaluation()
