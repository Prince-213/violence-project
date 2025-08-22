# Violence Detection in Video Feeds

![Project Screenshot](Screenshot%202025-08-22%20041142.png)

## Overview

This Python project analyzes video feeds in real-time to detect violent actions and sequences. Using deep learning models, the system processes video input, identifies potential violent behaviors, and generates detailed reports of detected incidents.

## Features

- 🎥 Real-time violence detection in video streams
- 🤖 Deep learning model integration (TensorFlow/Keras)
- 📊 Detailed incident reporting with timestamps
- ⚡ Fast processing with optimized model inference
- 📝 Text output of detection results
- 🔄 Support for various video formats

## Project Structure

```
violence-detection/
├── __pycache__/          # Python bytecode cache
├── model/                # Model directory
├── video/                # Sample videos directory
├── .gitignore           # Git ignore rules
├── check.py             # Validation/checking script
├── class.ipynb          # Jupyter notebook for classification
├── demo1.mp4            # Demonstration video
├── main.py              # Main application entry point
├── model.h5             # Trained model weights
├── my_model.h5          # Alternative model file
├── output.txt           # Sample output report
├── require.txt          # Requirements/dependencies
├── test.ipynb           # Testing notebook
└── test.py              # Testing script
```

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r require.txt
```

3. Ensure you have the necessary deep learning libraries:
- TensorFlow/Keras
- OpenCV
- NumPy
- Other dependencies listed in require.txt

## Usage

### Running the Main Application
```bash
python main.py --input path/to/video.mp4 --output report.txt
```

### Command Line Arguments
- `--input`: Path to input video file
- `--output`: Path to output report file (optional)
- `--threshold`: Confidence threshold for detection (default: 0.7)
- `--display`: Show video output with detection overlay (default: True)

### Example
```bash
python main.py --input video/demo1.mp4 --output violence_report.txt --threshold 0.8
```

## Model Information

The project uses a deep learning model (`model.h5`/`my_model.h5`) trained to recognize violent actions in video sequences. The model architecture is based on:

- Convolutional Neural Networks (CNNs) for spatial feature extraction
- Recurrent layers (LSTM/GRU) for temporal sequence analysis
- Transfer learning from pre-trained models (if applicable)

## Output Reports

The system generates detailed reports in `output.txt` format containing:
- Timestamps of detected violent incidents
- Confidence scores for each detection
- Duration of violent sequences
- Frame numbers where violence was detected

## Performance

- Real-time processing capabilities (depending on hardware)
- Adjustable confidence thresholds
- Support for various video resolutions
- Efficient memory management for long video sequences

## Customization

### Training Your Own Model
1. Use the Jupyter notebooks (`class.ipynb`, `test.ipynb`)
2. Prepare labeled training data
3. Adjust model architecture as needed
4. Train and export to `.h5` format

### Modifying Detection Parameters
Edit the configuration in `main.py` to adjust:
- Detection sensitivity
- Output format
- Processing frame rate
- Report generation settings

## Applications

- Security surveillance monitoring
- Content moderation for video platforms
- Public safety in crowded areas
- Sports activity analysis
- Educational institution monitoring

## Limitations

- Performance depends on video quality and lighting conditions
- May have false positives in certain scenarios
- Requires substantial computational resources for real-time processing
- Model accuracy depends on training data diversity

## Future Enhancements

- Real-time alert system integration
- Cloud-based processing for multiple streams
- Mobile application version
- Enhanced model with more violence categories
- Integration with existing security systems

## Support

For technical issues:
1. Check that all dependencies are properly installed
2. Ensure video files are in supported formats
3. Verify model files are in the correct directory
4. Consult the test scripts for functionality verification

## License

This project is intended for research and educational purposes. Commercial use may require additional licensing depending on application.

---

**Important Note**: This violence detection system should be used responsibly and in compliance with privacy laws and regulations. Always ensure proper authorization when monitoring video feeds.
