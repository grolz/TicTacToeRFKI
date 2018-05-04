package de.hbo.dl4j.ttt.classifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import de.hbo.tictactoe.TTTGame.Prompter;
import de.hbo.tictactoe.TTTGame.TTTPlayer;
import de.hbo.tictactoe.TicTacToe;

public abstract class AbstractTTTClassifierPlayer implements TTTPlayer {
	
	protected final MultiLayerNetwork model;
	
	protected AbstractTTTClassifierPlayer(final MultiLayerNetwork model) {this.model = model;}
	
	@Override public void draw(TicTacToe ttt, Prompter prompter) {this.draw(ttt);}
		
	
	private void draw(TicTacToe ttt) {

    	// prepare input for NN
		float[] features = new float[10];
		int numberOfMovesMade = 0;
		for (int i = 0; i < ttt.getBoard().length; i++) {
			features[i] = ttt.getBoard()[i];
			if (ttt.getBoard()[i] != 0) numberOfMovesMade++; 
		}
		features[9] = numberOfMovesMade==0? numberOfMovesMade : -ttt.getNext();

    	// get NN-Output
		double[] prediction = model.output(new NDArray(features),false).data().asDouble();
		
		// console NN-Output with available moves
		List<Integer> moves = new ArrayList<>();
		for (int i = 0; i < prediction.length; i++) if (prediction[i] > .4) {
			String m = Integer.toBinaryString(i);
			while(m.length() < 9) m = "0" +m;
			for (int j = 0; j < 9; j++) if (m.charAt(j)=='1') moves.add(new Integer(j));
			break;
		}
		
		int[] available = ttt.getAvailableMoves();
		
		moves.removeIf(each -> {for (int i = 0; i < available.length; i++) if (each.intValue()==available[i]) return false; return true;});
		if (moves.isEmpty()) for (int i = 0; i < available.length; i++) moves.add(available[i]);
    	
		// actually make a move
		ttt.move(moves.get(new Random().nextInt(moves.size())));
	}

	public static class Model1ComputerPlayer extends AbstractTTTClassifierPlayer {
		
		protected static MultiLayerNetwork model1;
		
		static {try {
			model1 = ModelSerializer.restoreMultiLayerNetwork(Model1ComputerPlayer.class.getClassLoader().getResourceAsStream("ttt_model_1.zip"));
		} catch (IOException e) {
			e.printStackTrace();
		}}
		
		public Model1ComputerPlayer() {super(model1);}
	}
	
	public static class Model9995ComputerPlayer extends AbstractTTTClassifierPlayer {
	
		protected static MultiLayerNetwork model9995;
		
		static {try {
			model9995 = ModelSerializer.restoreMultiLayerNetwork(Model1ComputerPlayer.class.getClassLoader().getResourceAsStream("ttt_model_.9995.zip"));
		} catch (IOException e) {
			e.printStackTrace();
		}}
		
		public Model9995ComputerPlayer() {super(model9995);}
	}
}
