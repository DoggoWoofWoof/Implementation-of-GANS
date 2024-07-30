# Clear upload folder at startup
clear_upload_folder()

# Register function to clear upload folder at shutdown
atexit.register(clear_upload_folder)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'sketch' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Both sketch and target files are required'}), 400
        
        sketch_file = request.files['sketch']
        target_file = request.files['target']
        
        if sketch_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if sketch_file and allowed_file(sketch_file.filename) and target_file and allowed_file(target_file.filename):
            sketch_filename = secure_filename(sketch_file.filename)
            target_filename = secure_filename(target_file.filename)
            
            sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch_filename)
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
            
            # Resize and save sketch
            sketch_img = Image.open(sketch_file)
            sketch_img = sketch_img.resize((128, 128), Image.LANCZOS)
            sketch_img.save(sketch_path)
            
            # Resize and save target
            target_img = Image.open(target_file)
            target_img = target_img.resize((128, 128), Image.LANCZOS)
            target_img.save(target_path)
            
            # Process images
            img, norm_img = load_and_preprocess_image(sketch_path)
            g_img = generate_image(g_model, norm_img)
            
            # Compute scores
            target = cv2.imread(target_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            l2 = compute_l2(g_img, target)
            ssim_score = compute_ssim(g_img, target, win_size=7, channel_axis=-1)
            
            # Save generated image
            generated_filename = 'generated_' + sketch_filename
            generated_path = os.path.join(app.config['UPLOAD_FOLDER'], generated_filename)
            plt.imsave(generated_path, g_img.astype(np.uint8))
            