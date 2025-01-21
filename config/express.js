const express = require("express");
const app = express();
const http = require("http").Server(app);
const io = require("socket.io")(http);
const express_config = require("./config/express.js"); // Import the express config
const port = 3000;

// Initialize express with the config
express_config.init(app);

var cameraArray = {};

app.get("/", (req, res) => {
  res.render("index", {}); // Ensure you have the 'views/index.ejs' file or modify this
});

io.on("connection", (socket) => {
  socket.on("jpgstream_server", (msg) => {
    io.to("webusers").emit("jpgstream_client", msg);
  });

  socket.on("webuser", (msg) => {
    socket.join("webusers");
  });
});

http.listen(port, () => {
  console.log(`App listening at http://localhost:${port}`);
});
