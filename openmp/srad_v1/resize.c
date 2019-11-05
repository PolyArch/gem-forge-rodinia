//====================================================================================================100
//====================================================================================================100
//	RESIZE FUNCTION
//====================================================================================================100
//====================================================================================================100

void resize(fp *input, int input_rows, int input_cols, fp *output,
            int output_rows, int output_cols, int major) {

  //================================================================================80
  //	VARIABLES
  //================================================================================80

  int i, j;
  int i2, j2;

  //================================================================================80
  //	COMPUTATION
  //================================================================================80

  //============================================================60
  //	ROW MAJOR
  //============================================================60

  if (major == 0) { // do if data is saved row major

    for (i = 0, i2 = 0; i < output_rows; i++, i2++) {
      if (i2 >= input_rows) {
        i2 = i2 - input_rows;
      }
      for (j = 0, j2 = 0; j < output_cols; j++, j2++) {
        if (j2 >= input_cols) {
          j2 = j2 - input_cols;
        }
        output[i * output_cols + j] = input[i2 * input_cols + j2];
      }
    }

  }

  //============================================================60
  //	COLUMN MAJOR
  //============================================================60

  else { // do if data is saved column major

    for (j = 0, j2 = 0; j < output_cols; j++, j2++) {
      if (j2 >= input_cols) {
        j2 = j2 - input_cols;
      }
      for (i = 0, i2 = 0; i < output_rows; i++, i2++) {
        if (i2 >= input_rows) {
          i2 = i2 - input_rows;
        }
        output[j * output_rows + i] = input[j2 * input_rows + i2];
      }
    }
  }
}
