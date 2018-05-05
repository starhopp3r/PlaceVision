//
//  ViewController.swift
//  PlaceVision
//
//  Created by Nikhil Raghavendra on 4/5/18.
//  Copyright © 2018 Nikhil Raghavendra. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    // MARK: Properties
    @IBOutlet weak var imagePicked: UIImageView!
    @IBOutlet weak var predictedLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override var shouldAutorotate: Bool {
        // Disable auto rotation
        return false
    }
    
    // MARK: UIImagePickerControllerDelegate
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        // Animate the cancellation of image picking when transiting
        // back to the original view.
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        // Convert the selected image to UIImage if unconvertable, return nil and fire the
        // guard case. The guard case returns a fatalError with an error message.
        guard let selectedImage = info[UIImagePickerControllerOriginalImage] as? UIImage else {
            fatalError("Error: \(info)")
        }
        // Set the UIImageView's image as the selected image
        imagePicked.image = selectedImage
        // Convert the UIImage to a CIImage. If unconvertable, return a fatalError.
        guard let pickedImage = CIImage(image: selectedImage) else {
            fatalError("couldn't convert UIImage to CIImage")
        }
        // Dismiss the current view and go back to the previous view
        dismiss(animated: true, completion: nil)
        // Detect the place
        detectPlace(image: pickedImage)
    }
    
    // MARK: Actions
    // Called when user taps the UIImageView
    @IBAction func selectImage(_ sender: Any) {
        // Initialize imagePickerController
        let imagePickerController = UIImagePickerController()
        // Restrict photo selection to photo library
        imagePickerController.sourceType = .photoLibrary
        // Notify ViewController when an image is picked
        imagePickerController.delegate = self
        // Display the imagePickerController over the current view and animate
        // the transisition of the view's overlay action.
        present(imagePickerController, animated: true, completion: nil)
    }
}

// MARK: ML Methods
extension ViewController {
    // detectPlace takes in an image of type CIImage
    func detectPlace(image: CIImage) {
        // Change the predictedLabel text as soon as image is passed.
        predictedLabel.text = "Detecting..."
        // Load the ML model into VNCoreMLModel: a container for a Core ML model used with Vision requests.
        // If the process fails, returns nil and causes the guard to be executed which throws a fatalError.
        guard let model = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
            fatalError("Error loading the model...")
        }
        // Create a vision request with the completion handler. request is an image analysis request that uses a
        // Core ML model to process images. VNCoreMLRequest is an image analysis request that uses a Core ML model
        // to do the work. Its completion handler receives request and error objects.
        let request = VNCoreMLRequest(model: model, completionHandler: { (request, error) in
            // Convert the results into a VNClassificationObservation. If unconvertable, return nil. If nil
            // is returned, the guard is executed and throws a fatalError.
            guard let results = request.results as? [VNClassificationObservation],
            // The first element of the collection, result, is assigned to the topResult constant. If unassignable
            // it throws a fatalError.
            let topResult = results.first else {
                fatalError("Unexpected result type from VNCoreMLRequest")
            }
            // Update UI on main queue. async lets the calling queue move on without waiting until the dispatched
            // block is executed. It returns control on the current queue right after task has been sent to be
            // performed on the different queue. It doesn't wait until the task is finished. It doesn't block the
            // queue. Closures can cause retain cycles for a single reason: by default, they have a strong reference
            // to the object that uses them. [weak self] causes the closure to have a weak refernce to the object
            // that uses them. We unwrap the value of the optional type, ViewController? using self?.
            DispatchQueue.main.async {[weak self] in
                // Change the predictedLabel's text to the classification result.
                self?.predictedLabel.text = "\(topResult.identifier)"
            }
        })
        
        // VNImageRequestHandler is the standard Vision framework request handler; it isn’t specific to Core ML
        // models. We give it the image that came into detectPlace(image:) as an argument. And then we run the
        // handler by calling its perform method, passing an array of requests. In this case, we have only one
        // request. The perform method throws an error, so we wrap it in a try-catch. The qos (quality of service
        // aka priority) is set to .userInteractive, which sets the task performing priority to the highest level.
        // async lets the calling queue move on without waiting until the dispatched block is executed.
        let handler = VNImageRequestHandler(ciImage: image)
        // Run the Core ML GoogLeNetPlaces classifier on the global dispatch queue.
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                print(error)
            }
        }
    }
}

