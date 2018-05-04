package de.hbo.dl4j.ttt;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

import de.hbo.dl4j.ttt.classifier.AbstractTTTClassifierPlayer.Model1ComputerPlayer;
import de.hbo.dl4j.ttt.classifier.AbstractTTTClassifierPlayer.Model9995ComputerPlayer;
import de.hbo.tictactoe.TTTGame;
import de.hbo.tictactoe.TicTacToe;

public class TTTToy {

    public static void main(String[] args) throws Exception {

    	final TTTGame.TTTPlayer[] players = new TTTGame.TTTPlayer[] {new Model9995ComputerPlayer(), new Model9995ComputerPlayer()};
    	int[] results = new int[] {0, 0, 0};
    	
    	System.out.println();
    	System.out.println();
    	
    	System.out.println("standing:");
    	for (int i = 0; i < 100; i++) try (PrintStream ps = new PrintStream(new OutputStream() {@Override public void write(int b) throws IOException {}})) {
    		TicTacToe ttt = TTTGame.play(players, (results[0] + results[1] + results[2])%2, null, ps); 
    		switch (ttt.isWin()) {
	    		case 0: results[2] = results[2]+1; break;
	    		case 1: 
	    			//System.out.println(ttt);
	    			results[(results[0] + results[1] + results[2])%2]+=1;
	    			break;
	    		case -1: 
	    			//System.out.println(ttt);
	    			results[(results[0] + results[1] + results[2])%2==0? 1 : 0] += 1;
	    	}
    		
    		System.out.print("(" + ttt.isWin() + ") " + results[0] + " : " + results[2] + " : " + results[1] + "\r");
    	}
    	
    	System.out.println();
        System.out.println("... done.");
    }
}




























