﻿using System;
using System.Collections.Generic;
using System.IO;
using Accord.Math;

    class MNIST
    {
        public static List<DigitImage> ReadMNIST(bool Visualize)
        {
        var trainingData = new List<DigitImage>();
            try
            {
                Console.Write("\nLoading MNIST...");
                FileStream ifsLabels =
                 new FileStream(@"MNIST/train-labels.idx1-ubyte",
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(@"MNIST/train-images.idx3-ubyte",
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                byte[][] pixels = new byte[28][];

    
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];



                // each test image
                for (int di = 0; di < 60000; ++di)
                {
                if (di % 1000 == 0)
                    Console.Write(".");
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();
           
                    DigitImage dImage =
                      new DigitImage(pixels, lbl);

                dImage.pixelsDbl = dImage.ToDouble();

                if (Visualize)
                {
                    Console.WriteLine(dImage.ToString());
                    Console.WriteLine(dImage.ToDouble().ToString<double>());
                }

                trainingData.Add(dImage);
                   // Console.ReadLine();
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

            Console.WriteLine("\nLoaded " + trainingData.Count + " Elements");
                Console.WriteLine("\nCompleted\n");


            return trainingData;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            return null;
            }
     
        } // Main
    } // Program

    public class DigitImage
    {    
        public byte[][] pixels;
        public double[][] pixelsDbl;
        public string label;
        public double[] labelDbl;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

        this.label = label.ToString();
        labelDbl = GenerateLabel(this.label);
        }

    public double[] GenerateLabel(string Label){
        int labelInt = int.Parse(Label);

        var outputs = new double[][]{
            new double[]{1,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0},
            new double[]{0,1,0,0,0,0,0,0,0,0, 0, 0, 0, 0},
            new double[]{0,0,1,0,0,0,0,0,0,0, 0, 0, 0, 0},
            new double[]{0,0,0,1,0,0,0,0,0,0, 0, 0, 0, 0},
            new double[]{0,0,0,0,1,0,0,0,0,0, 0, 0, 0, 0},
            new double[]{0,0,0,0,0,1,0,0,0,0, 0, 0, 0, 0},
            new double[]{0,0,0,0,0,0,1,0,0,0, 0, 0, 0, 0},
            new double[]{0,0,0,0,0,0,0,1,0,0, 0, 0, 0, 0},
            new double[]{0,0,0,0,0,0,0,0,1,0, 0, 0, 0, 0},
            new double[]{0,0,0,0,0,0,0,0,0,1, 0, 0, 0, 0},
        };

        return outputs[labelInt];
    }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString

    public double[][] ToDouble(){
        
        var val = new double[28][];
        for (int i = 0; i < val.Length; ++i)
            val[i] = new double[28];

        for (int i = 0; i < 28; ++i)
        {
            for (int j = 0; j < 28; ++j)
            {
                val[i][j] = pixels[i][j]/255.0;
            }
      
        }

        return val;
    }


    }