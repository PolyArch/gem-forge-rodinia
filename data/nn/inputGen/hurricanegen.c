/*
 * hurricanegen.c
 * Original author unknown
 * Modified by Sam Kauffman - University of Virginia
 *
 * Generates datasets of "hurricanes" to be used by Rodinia's Nearest Neighbor
 * (nn) Also generates lists of the files in the dataset. These lists are passed
 * to nn.
 *
 * Usage: hurricanegen <num_hurricanes> <num_files>
 * The number of hurricanes should be a multiple of both 1024 and the number of
 * files.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__attribute__((packed)) struct Record {
  int year;
  int month;
  int date;
  int hour;
  int num;
  int speed;
  int press;
  float lat;
  float lon;
  char name[10];
};

// 641986 gets you ~30 MB of data
int main(int argc, char **argv) {
  int hours[4] = {0, 6, 12, 18};
  char names[21][10] = {
      "ALBERTO", "BERYL", "CHRIS",  "DEBBY", "ERNESTO", "FLORENCE", "GORDON",
      "HELENE",  "ISAAC", "JOYCE",  "KIRK",  "LESLIE",  "MICHAEL",  "NADINE",
      "OSCAR",   "PATTY", "RAFAEL", "SANDY", "TONY",    "VALERIE",  "WILLIAM"};

  if (argc < 3) {
    fprintf(stderr,
            "Error: Enter a number of hurricanes and a number of files.\n");
    fprintf(stderr, "The number of hurricanes should be a multiple of both "
                    "1024\nand the number of files.\n");
    exit(0);
  }

  int total_canes = atoi(argv[1]);
  int num_files = atoi(argv[2]);

  total_canes =
      ((total_canes + 1023) / 1024) * 1024; // round up to multiple of 1024
  int canes =
      (total_canes + num_files - 1) / num_files; // round up (ceiling division)
  total_canes = canes * num_files;

  srand(time(NULL));

  for (int j = 0; j < num_files; j++) {
    char fname[30];
    char bin_fname[31];
    if (num_files == 1) {
      sprintf(fname, "cane%dk.db", total_canes / 1024);
      sprintf(bin_fname, "cane%dk.db.data", total_canes / 1024);
    } else {
      sprintf(fname, "cane%dk_%d_%d.db", total_canes / 1024, num_files, j);
      sprintf(bin_fname, "cane%dk_%d_%d.db.data", total_canes / 1024, num_files,
              j);
    }

    FILE *fp = fopen(fname, "w");
    FILE *bin_fp = fopen(bin_fname, "wb");
    if (!fp || !bin_fp) {
      fprintf(stderr, "Failed to open output file '%s'!\n", fname);
      return -1;
    }

    for (int i = 0; i < canes; i++) {
      struct Record record;
      record.year = 1950 + rand() % 55;
      record.month = 1 + rand() % 12;
      record.date = 1 + rand() % 28;
      record.hour = hours[rand() % 4];
      record.num = 1 + rand() % 28;
      strcpy(record.name, names[rand() % 21]);
      record.lat =
          ((float)(7 + rand() % 63)) + ((float)rand() / (float)0x7fffffff);
      record.lon =
          ((float)(rand() % 358)) + ((float)rand() / (float)0x7fffffff);
      record.speed = 10 + rand() % 155;
      record.press = rand() % 900;

      fprintf(fp, "%4d %2d %2d %2d %2d %-9s %5.1f %5.1f %4d %4d\n", record.year,
              record.month, record.date, record.hour, record.num, record.name,
              record.lat, record.lon, record.speed, record.press);
      fwrite(&record, sizeof(record), 1, bin_fp);
    }

    fclose(fp);
    fclose(bin_fp);
  }
  printf("Generated %d hurricanes in %d file(s).\n", total_canes, num_files);

  if (num_files == 1) {
    char fname[30];
    char bin_fname[30];
    sprintf(fname, "list%dk.txt", total_canes / 1024);
    sprintf(bin_fname, "list%dk.data.txt", total_canes / 1024);
    FILE *fp = fopen(fname, "w");
    FILE *bin_fp = fopen(bin_fname, "w");
    fprintf(fp, "../../data/nn/cane%dk.db\n", total_canes / 1024);
    fprintf(bin_fp, "../../data/nn/cane%dk.db.data\n", total_canes / 1024);
    fclose(fp);
    fclose(bin_fp);
    printf("File list written to %s.\n", fname);
  } else {
    char fname[30];
    char bin_fname[30];
    sprintf(fname, "list%dk_%d.txt", total_canes / 1024, num_files);
    sprintf(bin_fname, "list%dk_%d.data.txt", total_canes / 1024, num_files);
    FILE *fp = fopen(fname, "w");
    FILE *bin_fp = fopen(bin_fname, "w");
    for (int i = 0; i < num_files; i++) {
      fprintf(fp, "../../data/nn/cane%dk_%d_%d.db\n", total_canes / 1024,
              num_files, i);
      fprintf(bin_fp, "../../data/nn/cane%dk_%d_%d.db.data\n",
              total_canes / 1024, num_files, i);
    }
    fclose(fp);
    fclose(bin_fp);
    printf("File list written to %s.\n", fname);
  }

  return 0;
}
