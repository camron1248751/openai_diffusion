import React, { useState, useEffect } from "react";
import { io } from "socket.io-client";
import "./index.css";
import "./custom.css";
import "animate.css";
import Cropper from "react-cropper";
import "cropperjs/dist/cropper.css";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [croppedImage, setCroppedImage] = useState(null);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [progress, setProgress] = useState("");
  const [socket, setSocket] = useState(null);
  const [cropper, setCropper] = useState(null);

  useEffect(() => {
    const newSocket = io("http://localhost:5000");
    setSocket(newSocket);

    newSocket.on("progress_update", (message) => {
      setProgress(message);
    });

    return () => newSocket.close();
  }, []);

  const handleChange = (e) => {
    const fileReader = new FileReader();
    fileReader.onload = () => {
      setFile(fileReader.result);
    };
    fileReader.readAsDataURL(e.target.files[0]);
    setFileName(e.target.files[0].name);
  };

  const handleCropDone = () => {
    setCroppedImage(cropper.getCroppedCanvas().toDataURL());
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!croppedImage) return;

    const formData = new FormData();
    const blob = await (await fetch(croppedImage)).blob();

    formData.append("uploadedImage", blob, "cropped_image.png");

    const response = await fetch("http://localhost:5000/generate", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      setGeneratedImages(data.generatedFilenames);
      console.log("Generated images:", data.generatedFilenames);
    } else {
      alert("Error generating images. Please try again.");
    }
  };

  return (
    <div className="min-h-screen animated-gradient flex items-center justify-center">
      <div className="animate__animated animate__fadeIn container mx-auto px-4 py-5 bg-white rounded-lg shadow-md w-full max-w-md">
        <h1 className="animate__animated animate__fadeInDown text-4xl font-bold mb-5 text-center text-purple-800">
          Image Generation
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <label
            htmlFor="file"
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded w-full cursor-pointer inline-flex items-center justify-center"
          >
            <span>Choose file</span>
            <input
              type="file"
              id="file"
              accept=".png"
              onChange={handleChange}
              className="hidden"
            />
          </label>
          {fileName && <p className="text-sm text-gray-500 text-center">{fileName}</p>}
          {file && !croppedImage && (
            <>
              <Cropper
                src={file}
                style={{ height: 400, width: "100%" }}
                initialAspectRatio={1}
                aspectRatio={1}
                guides={false}
                onInitialized={(instance) => setCropper(instance)}
                />
                <button
                  type="button"
                  onClick={handleCropDone}
                  className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded w-full my-4"
                >
                  Done
                </button>
              </>
            )}
            {croppedImage && (
              <>
                <img src={croppedImage} alt="Cropped" className="w-full h-auto mb-4" />
                <button
                  type="button"
                  onClick={() => setCroppedImage(null)}
                  className="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-2 px-4 rounded w-full mb-4"
                >
                  Edit Crop
                </button>
              </>
            )}
            <button
              type="submit"
              className="animate__animated animate__pulse animate__infinite animate__slow bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded w-full"
            >
              Generate Images
            </button>
          </form>
          <br />
          <div className="text-lg text-center">{progress}</div>
          <br />
          <div className="grid grid-cols-2 gap-4">
            {generatedImages.map((filename, index) => (
              <img
                key={index}
                src={`http://localhost:5000/uploads/${filename}`}
                alt="Generated"
                className="animate__animated animate__fadeInUp w-full h-auto"
              />
            ))}
          </div>
        </div>
      </div>
  );
}
export default App      
