const express = require("express");
const http = require("http");
const socketIo = require("socket.io");
const fs = require("fs");
const path = require("path");

// Initialize express app and HTTP server
const app = express();
const server = http.createServer(app);

// Initialize Socket.IO
const io = socketIo(server);

// Serve static files (optional)
app.use(express.static("public"));

// Listen for incoming Socket.IO connections
io.on("connection", (socket) => {
  console.log("A user connected");

  // Listen for 'jpgstream_server' event from ESP32-CAM
  socket.on("jpgstream_server", (data) => {
    console.log("Received image data from:", data.hostname);

    // Log the base64 encoded picture (for debugging)
    console.log("Base64 image:", data.picture);

    // Define the path where the image will be saved
    const imagesFolderPath = path.join(__dirname, "images");
    const imageFilePath = path.join(imagesFolderPath, "latest_image.jpg"); // Static filename to overwrite

    // Ensure the images folder exists
    if (!fs.existsSync(imagesFolderPath)) {
      fs.mkdirSync(imagesFolderPath, { recursive: true });
    }

    // Base64 decoding: Remove the data URL prefix if present
    const base64Data = data.picture.replace(/^data:image\/jpeg;base64,/, ""); // Remove Base64 prefix

    // Write the image to the "latest_image.jpg" file, replacing the old one
    fs.writeFile(imageFilePath, base64Data, "base64", (err) => {
      if (err) {
        console.error("Error saving image:", err);
      } else {
        console.log("Image saved successfully as latest_image.jpg!");
      }
    });
  });

  // Handle disconnection
  socket.on("disconnect", () => {
    console.log("User disconnected");
  });
});

// Start the server on port 3000 (or any port you prefer)
const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
