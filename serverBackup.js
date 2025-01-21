const express = require("express");
const http = require("http");
const socketIo = require("socket.io");

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

    // Log the base64 encoded picture
    console.log("Base64 image:", data.picture);

    // You can save the image or process it as needed here
    // Example: Save the image to a file (you can use 'fs' module for that)

    // If you need to handle the base64 string to convert back to an image, do this:
    const fs = require("fs");
    const base64Data = data.picture.replace(/^data:image\/jpeg;base64,/, ""); // Remove Base64 prefix
    fs.writeFile(
      `image_from_${data.hostname}.jpg`,
      base64Data,
      "base64",
      (err) => {
        if (err) {
          console.error("Error saving image:", err);
        } else {
          console.log("Image saved successfully!");
        }
      }
    );
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
