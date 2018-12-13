using System;
using System.Collections.Generic;

namespace Perceptron
{
    public class Data
    {
        public Dictionary<int, int> TrainingData;

        public void GenerateTrainingData(int Amount)
        {
            Console.WriteLine("Generating Data");
            for (int i = 0; i < Amount; i++)
            {
                TrainingData.Add(i, i%2);
                Console.WriteLine(String.Format("{0}: {1}", i, i%2));
            }
        }

        public Data(int Amount)
        {
            TrainingData = new Dictionary<int, int>();
            GenerateTrainingData(Amount);
        }


    }
}
