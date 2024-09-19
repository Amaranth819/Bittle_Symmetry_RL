#define BITTLE
//number of skills: 56


const int8_t wh[] PROGMEM = { 
-3, 0, -30, 1,
 0, 1, 2, 
   57, -20,  20,   0,   0,   0,  40,  40,  33,  20,  62,  59,  46,  70, -15, -15,	48, 0, 0, 0,

};
const int8_t zz[] PROGMEM = { 
-1, 0, 0, 1,
 0, 0, 0, 
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,	 4, 0, 0, 0,
};
  const char* skillNameWithType[]={"whI","zzI",};
#if !defined(MAIN_SKETCH) || !defined(I2C_EEPROM)
		//if it's not the main sketch to save data or there's no external EEPROM, 
		//the list should always contain all information.
  const int8_t* progmemPointer[] = { wh, zz, };
#else	//only need to know the pointers to newbilities, because the intuitions have been saved onto external EEPROM,
	//while the newbilities on progmem are assigned to new addresses
  const int8_t* progmemPointer[] = {};
#endif
//the total byte of instincts is 7677
//the maximal array size is 317 bytes of wkF. 
//Make sure to leave enough memory for SRAM to work properly. Any single skill should be smaller than 400 bytes for safety.
