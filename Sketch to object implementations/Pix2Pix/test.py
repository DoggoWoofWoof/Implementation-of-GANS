def test(input_image_path, output_image_path):
    gen = Generator().to(DEVICE)
    load_checkpoint(CHECKPOINT_GEN, gen)
    gen.eval()
    
    # Load and preprocess image
    input_image = np.array(Image.open(input_image_path).convert("RGB"))
    input_image = both_transform(image=input_image)["image"]
    input_image = transform_only_input(image=input_image)["image"]
    input_image = input_image.unsqueeze(0).to(DEVICE)
    
    # Generate image
    with torch.no_grad():
        generated_image = gen(input_image)
    
    # Save image
    save_image(generated_image * 0.5 + 0.5, output_image_path)  # Rescale to [0, 1]

input_image_path = "path/to/sketch.png"  # Path to your input sketch image
output_image_path = "path/to/generated_image.png"  # Path to save the generated image
test(input_image_path, output_image_path)
