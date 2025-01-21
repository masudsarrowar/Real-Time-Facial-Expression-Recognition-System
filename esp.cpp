#include "WiFi.h"
#include "esp_camera.h"
#include "base64.h"
#include <ArduinoJson.h>
#include <SocketIOclient.h>

// Pin definition for CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Replace with your network credentials
const char* hostname = "ESP32CAM";
const char* ssid = "UIU-STUDENT";
const char* password = "12345678";

SocketIOclient socketIO;
int port = 3000;  // Define the port (change if needed)

void socketIOEvent(socketIOmessageType_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case sIOtype_DISCONNECT:
            Serial.printf("[IOc] Disconnected!\n");
            break;
        case sIOtype_CONNECT:
            Serial.printf("[IOc] Connected to url: %s\n", payload);
            socketIO.send(sIOtype_CONNECT, "/");
            break;
        case sIOtype_EVENT:
            Serial.printf("[IOc] get event: %s\n", payload);
            break;
        case sIOtype_ACK:
            Serial.printf("[IOc] get ack: %u\n", length);
            break;
        case sIOtype_ERROR:
            Serial.printf("[IOc] get error: %u\n", length);
            break;
        case sIOtype_BINARY_EVENT:
            Serial.printf("[IOc] get binary: %u\n", length);
            break;
        case sIOtype_BINARY_ACK:
            Serial.printf("[IOc] get binary ack: %u\n", length);
            break;
    }
}

void setupCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    
    config.frame_size = FRAMESIZE_CIF;  // Change as needed (CIF, VGA, etc.)
    config.jpeg_quality = 10;
    config.fb_count = 2;
  
    // Init Camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
      Serial.printf("Camera init failed with error 0x%x", err);
      return;
    }
}

void setup() {
  Serial.begin(115200);
  
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..");
  }

  // Print ESP32 Local IP Address
  Serial.println(WiFi.localIP());

  setupCamera();
  
  // Connect to Socket.IO server
  socketIO.begin("10.15.17.129", 3000, "/socket.io/?EIO=4");
  // Set event handler
  socketIO.onEvent(socketIOEvent);
}

unsigned long messageTimestamp = 0;

void loop() {
    socketIO.loop();

    uint64_t now = millis();

    if(now - messageTimestamp > 10000) {  // Take picture every second (adjust time)
        messageTimestamp = now;

        camera_fb_t * fb = NULL;

        // Take Picture with Camera
        fb = esp_camera_fb_get();  
        if(!fb) {
          Serial.println("Camera capture failed");
          return;
        }
        
        // Encode the image to Base64
        String picture_encoded = base64::encode(fb->buf, fb->len);

        // Create JSON message for Socket.IO
        DynamicJsonDocument doc(15000);
        JsonArray array = doc.to<JsonArray>();
        
        // Add event name
        array.add("jpgstream_server");

        // Add payload (parameters) for the event
        JsonObject param1 = array.createNestedObject();
        param1["hostname"] = hostname;
        param1["picture"] = picture_encoded;  // Send the base64 string

        // Serialize JSON to String
        String output;
        serializeJson(doc, output);

        // Send the event
        socketIO.sendEVENT(output);
        Serial.println("Image sent");
        Serial.println(output);
        
        esp_camera_fb_return(fb);  // Return the frame buffer
    }
}
