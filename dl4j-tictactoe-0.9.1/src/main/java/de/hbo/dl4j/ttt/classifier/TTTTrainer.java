package de.hbo.dl4j.ttt.classifier;

import java.awt.Color;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import de.hbo.diagramm.diagramm2D.PunktDoubleDouble;
import de.hbo.diagramm.diagramm2D.StreckenzugDoubleDouble;
import de.hbo.dl4j.ttt.util.ChartDisplayDouble;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class TTTTrainer {


    public static void main(String[] args) throws Exception {

        final File data = new File(TTTTrainer.class.getClassLoader().getResource("tictactoe.txt").getPath());
        final int nEpochs = 10000;
        final int batchSize = 100;

    
    
        // ------------------------------------------------------------------------------------------------------------------
        // MODEL SETUP
        System.out.print("Setup model:");

        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(123)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(.01)
            .updater(new Nesterovs(0.9))
            .list()
            .layer(0, new DenseLayer.Builder().nIn(10).nOut(512).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
            .layer(1, new DenseLayer.Builder().nIn(512).nOut(512).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
            .layer(2, new DenseLayer.Builder().nIn(512).nOut(512).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
            .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).nIn(512).nOut(512).build())
            .pretrain(false)
            .backprop(true)
            .build()
		);
        model.init();

        System.out.println("done");
        
        
        // ------------------------------------------------------------------------------------------------------------------
        // DISPLAY
		
		ChartDisplayDouble display = new ChartDisplayDouble(0d, nEpochs, 0d, 1d, Color.BLACK, Color.GREEN);
		display.display();
		
		List<PunktDoubleDouble> accuracy = new ArrayList<>();
		List<PunktDoubleDouble> precision = new ArrayList<>();
        
        
        // ------------------------------------------------------------------------------------------------------------------
        // TRAINING
        
        System.out.println("Train model....");
        int epoch = 0;   
        Evaluation eval = null;
        double oldPrecision = Double.NaN;
        double oldAccuracy = Double.NaN;
        int numReps = 0;
     
        //Load data:
        try (TTTRecordReader rr = new TTTRecordReader()) {
        	
	        rr.initialize(new FileSplit(data));
	        DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,10,512);
	        
	        while (numReps < 10 && epoch < nEpochs) {
	        //for (int i = 0; i < 3; i++) {
	        	
	        	model.fit(iterator);
	        	
	        	try (TTTRecordReader rr2 = new TTTRecordReader()) {
	        		
	    	        rr2.initialize(new FileSplit(data));
			        eval = model.evaluate(new RecordReaderDataSetIterator(rr2,batchSize,10,512));
			        System.out.println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " - epoch " + trimInt(++epoch, 3) + " - precision: " + trimDouble(eval.precision(), 12) + ", accuracy: " + trimDouble(eval.accuracy(), 12) + ", score: " + model.score());
			        
			        if (eval.precision()==oldPrecision && eval.accuracy()==oldAccuracy) numReps++;
			        else numReps = 0;
			        
			        oldPrecision = eval.precision();
			        oldAccuracy = eval.accuracy();
	        	}
	        	
            	accuracy.add(new PunktDoubleDouble(null, null, epoch, oldAccuracy));
            	display.addOrReplace(new StreckenzugDoubleDouble("accuracy", Color.PINK, accuracy.toArray(new PunktDoubleDouble[0])), false);
            	precision.add(new PunktDoubleDouble(null, null, epoch, oldPrecision));
            	display.addOrReplace(new StreckenzugDoubleDouble("precision", Color.CYAN, precision.toArray(new PunktDoubleDouble[0])), true);
    		}

        }

        
//		List<Pair<INDArray, INDArray>> data = new ArrayList<>();
//
//        try (BufferedReader reader = new BufferedReader(new FileReader(trainingData))) {
//    		
//        	while (reader.ready()) {
//        		
//        		String line = reader.readLine();
//                int index = line.indexOf("|");
//                
//                String[] featureValues = line.substring(0, index).split(",", -1);
//            	float[] features = new float[featureValues.length];
//            	for (int i = 0; i < featureValues.length; i++) features[i] = Float.parseFloat(featureValues[i]);
//            	
//            	String labelString = line.substring(index+1);
//            	float[] label = new float[labelString.length()];
//            	for (int i = 0; i < labelString.length(); i++) label[i] = Float.parseFloat(labelString.substring(i, i+1));
//        		
//        		data.add(new Pair(features, label));
//        	}
//        }
//        	
//        	
//        while (numReps < 10 && epoque < 1000) {
//        //for (int i = 0; i < 3; i++) {
//        	
//        	model.fit(new INDArrayDataSetIterator(data, batchSize));
//        	
//        	try (TTTRecordReader rr2 = new TTTRecordReader()) {
//        		
//    	        rr2.initialize(new FileSplit(trainingData));
//		        eval = model.evaluate(new RecordReaderDataSetIterator(rr2,batchSize,10,512));
//		        System.out.println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " - epoque " + trimInt(++epoque, 3) + " - precision: " + trimDouble(eval.precision(), 12) + ", accuracy: " + trimDouble(eval.accuracy(), 12) + ", score: " + model.score());
//		        
//		        if (eval.precision()==oldPrecision && eval.accuracy()==oldAccuracy) numReps++;
//		        else numReps = 0;
//		        
//		        oldPrecision = eval.precision();
//		        oldAccuracy = eval.accuracy();
//        	}
//        }
        
        
        //ModelSerializer.writeModel(model, new File("src/main/resources/classification/ttt_model_512.zip"), true);
        
        System.out.println("... done.");
    }
    
    private static String trimDouble(double d, int length) {
    	String erg = Double.toString(d);
    	if (erg.length() > length) erg = erg.substring(0, length);
    	while (erg.length() < length) erg += (erg.indexOf(".") < 0)? "." : "0";
    	return erg;
    }
    
    private static String trimInt(int i, int length) {
    	String erg = Integer.toString(i);
    	while (erg.length() < length) erg = " " + erg;
    	return erg;
    }
    
    private static class EpoqueListener implements TrainingListener {

    	private boolean invoked = false;
    	private int epoque = 0;
    	private int iter = 0;
    	private int forward = 0;
    	private int backward = 0;
    	
		@Override public boolean invoked() {return this.invoked;}
		@Override public void invoke() {this.invoked = true;}

		@Override public void iterationDone(Model model, int iteration) {this.iter++;}

		@Override public void onEpochStart(Model model) {}

		@Override public void onEpochEnd(Model model) {System.out.println("epoque " + ++this.epoque + ": " + model.score());}

		@Override public void onForwardPass(Model model, List<INDArray> activations) {this.forward++;}

		@Override public void onForwardPass(Model model, Map<String, INDArray> activations) {}

		@Override public void onGradientCalculation(Model model) {}

		@Override public void onBackwardPass(Model model) {this.backward++;}
    	
    }
}


//04:15:40 - epoque   1 - precision: 0.1748813435, accuracy: 0.1748813435, score: 0.2000420093536377
//04:16:52 - epoque   2 - precision: 0.1748813435, accuracy: 0.1748813435, score: 0.26877972057887484
//04:18:07 - epoque   3 - precision: 0.1748813435, accuracy: 0.1748813435, score: 0.24986886978149414
//04:19:18 - epoque   4 - precision: 0.4256356818, accuracy: 0.1998904709, score: 0.2765899726322719
//04:20:30 - epoque   5 - precision: 0.4513922513, accuracy: 0.2404162102, score: 0.3071850367954799
//04:21:44 - epoque   6 - precision: 0.5684918618, accuracy: 0.2869660460, score: 0.39036379541669575
//04:22:57 - epoque   7 - precision: 0.6014835359, accuracy: 0.3201898503, score: 0.3913581371307373
//04:24:09 - epoque   8 - precision: 0.6167800410, accuracy: 0.3599853961, score: 0.3478809765407017
//04:25:14 - epoque   9 - precision: 0.5968898219, accuracy: 0.3389923329, score: 0.2993709700448172
//04:26:27 - epoque  10 - precision: 0.5141119140, accuracy: 0.3585250091, score: 0.24208761964525496
//04:27:43 - epoque  11 - precision: 0.4994005403, accuracy: 0.3634538152, score: 0.21376071657453263
//04:29:01 - epoque  12 - precision: 0.4951285529, accuracy: 0.3521358159, score: 0.15093064308166504
//04:30:21 - epoque  13 - precision: 0.5618580874, accuracy: 0.3338809784, score: 0.05078910504068647
//04:31:41 - epoque  14 - precision: 0.4945247071, accuracy: 0.3127053669, score: 0.004772155412605831
//04:32:52 - epoque  15 - precision: 0.4366315618, accuracy: 0.2995618838, score: 0.008284955152443476
//04:34:03 - epoque  16 - precision: 0.4717012393, accuracy: 0.3919313618, score: 0.052657629762377055
//04:35:14 - epoque  17 - precision: 0.4552846603, accuracy: 0.4362906170, score: 0.05085029772349766
//04:36:25 - epoque  18 - precision: 0.5354344162, accuracy: 0.4056224899, score: 0.09355242763246809
//04:37:35 - epoque  19 - precision: 0.4140742094, accuracy: 0.4432274552, score: 0.11831891536712646
//04:38:46 - epoque  20 - precision: 0.4872929148, accuracy: 0.4859437751, score: 1.466514996119908
//04:39:56 - epoque  21 - precision: 0.5653351927, accuracy: 0.3937568455, score: 0.6625783102852958
//04:41:07 - epoque  22 - precision: 0.4631764268, accuracy: 0.5719240598, score: 0.3302586759839739
//04:42:17 - epoque  23 - precision: 0.5467793292, accuracy: 0.6652062796, score: 0.03468571177550724
//04:43:27 - epoque  24 - precision: 0.6258691626, accuracy: 0.7102957283, score: 0.15810562883104598
//04:44:37 - epoque  25 - precision: 0.6426774696, accuracy: 0.6004016064, score: 7.16810513819967E-4
//04:45:47 - epoque  26 - precision: 0.7010921364, accuracy: 0.6670317634, score: 0.16932286534990584
//04:46:58 - epoque  27 - precision: 0.7708572409, accuracy: 0.7396860167, score: 0.002102500093834741
//04:48:10 - epoque  28 - precision: 0.7570475857, accuracy: 0.6239503468, score: 0.0011363700032234192
//04:49:24 - epoque  29 - precision: 0.8576491825, accuracy: 0.8243884629, score: 0.009707058114664895
//04:50:38 - epoque  30 - precision: 0.7908824429, accuracy: 0.6381891201, score: 0.01125836159501757
//04:51:54 - epoque  31 - precision: 0.8947787873, accuracy: 0.8420956553, score: 0.002375371754169464
//04:53:09 - epoque  32 - precision: 0.8013343915, accuracy: 0.6779846659, score: 0.1350877285003662
//04:54:22 - epoque  33 - precision: 0.8975928473, accuracy: 0.8169039795, score: 0.0014522003808191844
//04:55:36 - epoque  34 - precision: 0.8893935739, accuracy: 0.7745527564, score: 1.2602197654944445E-6
//04:56:50 - epoque  35 - precision: 0.9085234819, accuracy: 0.8183643665, score: 0.01329263299703598
//04:58:04 - epoque  36 - precision: 0.9255312575, accuracy: 0.8880978459, score: 0.0034846075411353794
//04:59:17 - epoque  37 - precision: 0.9382417834, accuracy: 0.9167579408, score: 0.0034704056701489855
//05:00:30 - epoque  38 - precision: 0.9227622614, accuracy: 0.9025191675, score: 1.9977810526532785E-4
//05:01:41 - epoque  39 - precision: 0.9576850064, accuracy: 0.9472435195, score: 6.139655514354152E-5
//05:03:11 - epoque  40 - precision: 0.9621770356, accuracy: 0.9614822928, score: 0.0028353847031082425
//05:04:38 - epoque  41 - precision: 0.9866224013, accuracy: 0.9771814530, score: NaN
//05:05:49 - epoque  42 - precision: 0.9911687248, accuracy: 0.9925155166, score: NaN





























