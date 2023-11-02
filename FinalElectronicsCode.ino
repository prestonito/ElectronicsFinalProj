// include the Library code:
#include <Wire.h>
#include <Adafruit_RGBLCDShield.h>
#include <utility/Adafruit_MCP23017.h>


// These #defines make it easy to set the backlight color
#define WHITE 0x7

// Creating objects that will be used throughout the code
String detection;
String xCoordStr;
float xCoord;
Adafruit_RGBLCDShield lcd = Adafruit_RGBLCDShield();


void setup() {
  // begin with 9600 baud rate
  Serial.begin(9600); 

  // set up the LCD's number of columns and rows
  lcd.begin(16, 2);
  lcd.clear();
  lcd.setBacklight(WHITE);

  // connects the pins to the corresponding motor functions
  pinMode(13,OUTPUT);   //right motors forward
  pinMode(12,OUTPUT);   //right motors reverse
  pinMode(11,OUTPUT);   //left motors reverse
  pinMode(10,OUTPUT);   //left motors forward
}

void loop() {
  // set the cursor to column 0, line 0 and clears it 
  lcd.setCursor(0, 0);
  lcd.clear();

  // if something is ready to be read, read it 
  if (Serial.available()) {
    detection = Serial.readString();  // reading string from Serial, which in this case comes from the Python file

    // moves/turns the car only if the only thing it's reading is "1 person"
    if (detection == "1 person") {    
      xCoordStr = Serial.readString();  // re-read the string from Serial now that we want the x-coordinate

      // filters through the readings; only takes strings that are clearly coordinates (starts with 0.)
      if (xCoordStr.substring(0,2) == "0.") {   
        xCoord = xCoordStr.substring(0, 5).toFloat();   // converts the x-coordinate string to a float

        // NOTE: The coordinates range from 0 to 1
        // turns the car left if the x-coordinate is to the left
        if (xCoord <= 0.4) {    
          lcd.print("Turning left");
          
          // turns left by turning the right motors forward and left motors backwards
          digitalWrite(13, HIGH);
          digitalWrite(11, HIGH);
        }

        // turns the car right if the x-coordinate is to the right
        else if (xCoord >= 0.6) {
          lcd.print("Turning right");

          // turns right by turning the left motors forward and right motors backwards
          digitalWrite(12, HIGH);
          digitalWrite(10, HIGH);
        }

        // if the x-coordinate of the human is in the range of 0.4-0.6, move forward
        else {
          lcd.print("Moving toward");
          lcd.setCursor(0, 1);
          lcd.print("person");
          lcd.setCursor(0, 0);
          digitalWrite(13, HIGH);
          digitalWrite(10, HIGH);
        }
      }
    }

    // if the detection is not "1 person" and not an x-coordinate, print what it sees to the lcd screen
    else if (detection.substring(0,2) != "0.") {
      lcd.print("I see");
      lcd.setCursor(0,1);
      lcd.print(detection);
    }
  }

  // if Serial.read is not available, just print "looking..." on the screen
  else {
    lcd.print("looking...");
  }
  
  // basically turn the motoros on for 250ms
  delay(250);

  // turn all motors off
  digitalWrite(13, LOW);
  digitalWrite(12, LOW);
  digitalWrite(11, LOW);
  digitalWrite(10, LOW);
  delay(500);
}
  

  

 


