package de.hbo.tictactoe;

import java.io.PrintStream;

public class TTTGame {
	
	public static TicTacToe play(final TTTPlayer[] players, final int startingPlayer, final Prompter prompter, final PrintStream out) {
		
		if (players == null || players.length != 2) throw new IllegalArgumentException("there must be two players.");
		if (startingPlayer != 0 && startingPlayer != 1) throw new IllegalArgumentException("starting player must be either 0 or 1.");
		
		int nextPlayer = startingPlayer;
		
		final TicTacToe ttt = new TicTacToe();
		while (!ttt.isFinal()) {
	    	out.println(ttt.toString());
	    	players[nextPlayer].draw(ttt, prompter);
	    	nextPlayer = nextPlayer==0? 1 : 0;
		}
		
		for (int i = 0; i < players.length; i++) players[i].feedback(ttt.isWin()*(i==startingPlayer? 1 : -1));

    	out.println();
    	out.println(ttt.toString());
    	out.println("result: " + (ttt.isWin()!=0? "win for '" + TicTacToe.getSymbol(ttt.isWin()) + "'" : (ttt.isDraw()? "draw" : "none so far")));
    	
    	return ttt;
	}

    
    public static interface Prompter {public String promptInput(String prompt);}
    
    public static interface TTTPlayer {
    	public void draw(TicTacToe ttt, Prompter prompter);
    	default void feedback(final int reward) {}; // nur fuer reinforcement player notwendig
	}
	
}
