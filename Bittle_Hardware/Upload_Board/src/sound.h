#define LENGTH_FACTOR 2
#define BASE_PITCH 1046.50
// tone: pause,1,  2,  3,  4,  5,  6,  7,  1,  2, 3, 4
// code: 0,    8, 10, 12, 13, 15, 17, 19, 20, 22,24,25

// byte melodyNormalBoot[] = {
//                           };
// byte melodyInit[] = {
//                     };
byte melodyNormalBoot[] = { //relative duration, 8 means 1/8 note length
                          };
byte melodyInit[] = {//relative duration, 8 means 1/8 note length
                    };
byte melodyLowBattery[] = {//relative duration, 8 means 1/8 note length
                          };
byte melody1[] = {
                 };

void beep(float note, float duration = 50, int pause = 0, byte repeat = 1 ) {
  for (byte r = 0; r < repeat; r++) {
    if (note)
      tone(BUZZER, BASE_PITCH * pow(1.05946, note));//tone(pin, frequency, duration) the duration doesn't work
    else
      delay(duration);
    delay(duration);
    noTone(BUZZER);
    delay(pause);
  }
}

void playMelody(int start) {
  byte len = (byte)EEPROM.read(start) / 2;
  for (int i = 0; i < len; i++)
    beep(EEPROM.read(start - 1 - i), 1000 / EEPROM.read(start - 1 - len - i));
}

void playTone(uint16_t tone1, uint16_t duration) {
  if (tone1 < 50 || tone1 > 15000) return; // these do not play on a piezo
  for (long i = 0; i < duration * 1000L; i += tone1 * 2) {
    digitalWrite(BUZZER, HIGH);
    delayMicroseconds(tone1);
    digitalWrite(BUZZER, LOW);
    delayMicroseconds(tone1);
  }
}

void meow(byte repeat = 2, byte duration = 10, byte startF = 210 + random() % 10,  byte endF = 220 + random() % 10) { // Bird chirp
  for (byte r = 0; r < repeat; r++)
    for (byte i = startF; i < endF; i++) {
      playTone(i, duration);
    }
}
