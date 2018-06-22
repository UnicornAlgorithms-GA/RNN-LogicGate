#pragma warning disable 0169
#pragma warning disable 0414

using System;
using System.Collections.Generic;
using System.Linq;
using GeneticLib.Generations;
using GeneticLib.Generations.InitialGeneration;
using GeneticLib.GeneticManager;
using GeneticLib.Genome;
using GeneticLib.Genome.NeuralGenomes;
using GeneticLib.Genome.NeuralGenomes.NetworkOperationBakers;
using GeneticLib.GenomeFactory;
using GeneticLib.GenomeFactory.GenomeProducer;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding.Crossover;
using GeneticLib.GenomeFactory.GenomeProducer.Reinsertion;
using GeneticLib.GenomeFactory.GenomeProducer.Selection;
using GeneticLib.GenomeFactory.Mutation;
using GeneticLib.GenomeFactory.Mutation.NeuralMutations;
using GeneticLib.Neurology;
using GeneticLib.Neurology.NeuralModels;
using GeneticLib.Neurology.Neurons;
using GeneticLib.Neurology.NeuronValueModifiers;
using GeneticLib.Randomness;
using GeneticLib.Utils.Graph;
using GeneticLib.Utils.NeuralUtils;

namespace RNN_LogicGate
{
    class Program
    {
        private static readonly string pyNeuralNetGraphDrawerPath =
            "../Submodules/MachineLearningPyGraphUtils/PyNeuralNetDrawer.py";
        private static readonly string pyFitnessGraphPath =
			"../Submodules/MachineLearningPyGraphUtils/DrawGraph.py";

        int genomesCount = 50;

        float singleSynapseMutChance = 0.2f;
        float singleSynapseMutValue = 1f;

        float allSynapsesMutChance = 0.1f;
        float allSynapsesMutChanceEach = 1f;
        float allSynapsesMutValue = 1f;

        float crossoverPart = 0.80f;
        float reinsertionPart = 0.2f;

        GeneticManagerClassic geneticManager;
        public static int maxIterations = 20000;
        public static bool targetReached = false;

		public int[][] samplePairs = new[]
		{
			new[] {0, 0},
			new[] {0, 1},
			new[] {1, 0},
			new[] {1, 1}
		};

		int[][] datasets;

        static void Main(string[] args)
        {
            GARandomManager.Random = new RandomClassic((int)DateTime.Now.Ticks);

            NeuralGenomeToJSONExtension.distBetweenNodes *= 5;
            NeuralGenomeToJSONExtension.randomPosTries = 10;
            NeuralGenomeToJSONExtension.xPadding = 0.03f;
            NeuralGenomeToJSONExtension.yPadding = 0.03f;

            NeuralNetDrawer.pyGraphDrawerPath = pyNeuralNetGraphDrawerPath;
            PyDrawGraph.pyGraphDrawerFilePath = pyFitnessGraphPath;

            var neuralNetDrawer = new NeuralNetDrawer(false);
            //var fitnessCollector = new GraphDataCollector();

            NeuralGenomeToJSONExtension.distBetweenNodes *= 5;
            NeuralGenomeToJSONExtension.randomPosTries = 10;

            var program = new Program();

            for (var i = 0; i < maxIterations; i++)
            {            
                program.Evaluate();
                var best = program.BestGenome() as NeuralGenome;
                //fitnessCollector.Tick(i, best.Fitness);
                Console.WriteLine(String.Format(
                    "{0}) Best:{1:0.00}",
                    i,
                    best.Fitness));

				if (targetReached)
				{
					foreach (var pair in program.samplePairs)
                    {
                        Console.WriteLine("{0} ^ {1} = {2} | {3:0.00}",
                                          pair[0], pair[1],
                                          pair[0] ^ pair[1],
						                  program.FeedRNNPair(best, pair));
                    }
					break;
				}

                program.Evolve();
            }         

            neuralNetDrawer.QueueNeuralNetJson((program.BestGenome() as NeuralGenome).ToJson(
                neuronRadius: 0.02f,
                maxWeight: 7,
                edgeWidth: 1f));
            //fitnessCollector.Draw();
        }
        
        public Program()
        {
			datasets = GenerateDatasets(100, GARandomManager.Random.Next(1, 50)).ToArray();
			//datasets = GenerateDatasets(100, 50).ToArray();

            var synapseTracker = new SynapseInnovNbTracker();

            var initialGenerationGenerator = new NeuralInitialGenerationCreatorBase(
                InitModel(),
				new FeedForwardOpBaker());

            //var selection = new EliteSelection();
            var selection = new RouletteWheelSelectionWithRepetion();
            var crossover = new OnePointCrossover(true);
            var breeding = new BreedingClassic(
                crossoverPart,
                1,
                selection,
                crossover,
                InitMutations()
            );

            var reinsertion = new ReinsertionFromSelection(
                reinsertionPart, 0, new EliteSelection());
            var producers = new IGenomeProducer[] { breeding, reinsertion };
            var genomeForge = new GenomeForge(producers);

            var generationManager = new GenerationManagerKeepLast();
            geneticManager = new GeneticManagerClassic(
                generationManager,
                initialGenerationGenerator,
                genomeForge,
                genomesCount
            );

            geneticManager.Init();
        }

        public void Evolve()
        {
            geneticManager.Evolve();
        }

        public void Evaluate()
        {
            var genomes = geneticManager.GenerationManager
                                        .CurrentGeneration
                                        .Genomes;

            foreach (var genome in genomes)
            {
                genome.Fitness = ComputeFitness(genome as NeuralGenome);
            }

            var orderedGenomes = genomes.OrderByDescending(g => g.Fitness)
                                        .ToArray();

            geneticManager.GenerationManager
                          .CurrentGeneration
                          .Genomes = orderedGenomes;
        }

        private float ComputeFitness(NeuralGenome genome)
        {
            genome.NetworkOperationBaker.BakeNetwork(genome);
            var fitness = 0d;
            
			foreach (var dataset in datasets)
			{
				genome.ResetNeuronsValues();

				int expectedResult = 0;
				foreach (var input in dataset)
				{
					expectedResult ^= input;
					genome.FeedNeuralNetwork(new float[] { input });               

					var delta = Math.Abs(expectedResult - genome.Outputs[0].Value);
                    fitness -= delta;
				}            
			}

            if (fitness >= -0.01f)
                targetReached = true;

            return (float)fitness;
        }

		private float FeedRNNPair(NeuralGenome genome, int[] values)
		{
			genome.ResetNeuronsValues();
			genome.FeedNeuralNetwork(new float[] { values[0] });
			genome.FeedNeuralNetwork(new float[] { values[1] });
			return genome.Outputs.First().Value;
		}

		private IEnumerable<int[]> GenerateDatasets(int size, int binaryStrLen)
		{
			for (int i = 0; i < size; i++)
			{
				yield return Enumerable.Range(0, binaryStrLen)
									   .Select(x => GARandomManager.Random.Next(0, 2))
									   .ToArray();
			}
		}

        private INeuralModel InitModel()
        {
            var model = new NeuralModelBase();
            model.defaultWeightInitializer = () => GARandomManager.NextFloat(-3, 3);
            model.WeightConstraints = new Tuple<float, float>(-20, 20);

            var bias = model.AddBiasNeuron();
            var layers = new[]
            {
                model.AddInputNeurons(1).ToArray(),

                model.AddNeurons(
					sampleNeuron: new Neuron(-1, ActivationFunctions.Gaussian),
                    count: 1
                ).ToArray(),
                
				model.AddOutputNeurons(1, ActivationFunctions.Sigmoid).ToArray()
            };

            model.ConnectBias(bias, layers.Skip(1));
            model.ConnectLayers(layers);

			foreach (var neuron in layers[1])
			{
				var mem = model.AddNeurons(
				    sampleNeuron: new MemoryNeuron(-1, neuron.InnovationNb),
					count: 1);
				model.AddConnection(mem[0].InnovationNb, neuron.InnovationNb);
			}
            
            return model;
        }

        private MutationManager InitMutations()
        {
            var result = new MutationManager();
            result.MutationEntries.Add(new MutationEntry(
                new SingleSynapseWeightMutation(() => singleSynapseMutValue),
                singleSynapseMutChance,
                EMutationType.Independent
            ));

            result.MutationEntries.Add(new MutationEntry(
                new SingleSynapseWeightMutation(() => singleSynapseMutValue * 5),
                singleSynapseMutChance / 5,
                EMutationType.Independent
            ));

            result.MutationEntries.Add(new MutationEntry(
                new AllSynapsesWeightMutation(
                    () => allSynapsesMutValue,
                    allSynapsesMutChanceEach),
                allSynapsesMutChance,
                EMutationType.Independent
            ));

            return result;
        }

        public IGenome BestGenome()
        {
            return geneticManager.GenerationManager
                                 .CurrentGeneration
                                 .Genomes
                                 .OrderByDescending(g => g.Fitness)
                                 .First();
        }
    }
}
