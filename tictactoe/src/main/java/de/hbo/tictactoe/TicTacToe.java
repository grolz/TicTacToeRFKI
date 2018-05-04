package de.hbo.tictactoe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TicTacToe {
	
	public static final int START = 1;
	
	private final static int[][] WINS = new int[][] {
		new int[] {0,1,2}, new int[] {3,4,5}, new int[] {6,7,8}, 	// waagrecht
		new int[] {0,3,6}, new int[] {1,4,7}, new int[] {2,5,8}, 	// senkrecht
		new int[] {0,4,8}, new int[] {2,4,6}						// diagonal
	};
	

	private final int[] board = new int[9];
	private int next = START;
	
	public TicTacToe() {Arrays.fill(board, 0);}
	
	public TicTacToe(int[] input) {
		this();
		for (int i = 0; i < this.board.length; i++) if (i < input.length && input[i] >= -1 && input[i] <= 1) {
			this.board[i] = input[i];
			if (input[i] != 0) next*=-1;
		}
	}
	
	public int[] getBoard() {return this.board.clone();}
	
	public int[] getAvailableMoves() {
		
		if (this.isDraw() || this.isWin()!= 0) return new int[0];
		
		List<Integer> moves = new ArrayList<>();
		for (int i = 0; i < board.length; i++) if (board[i]==0) moves.add(i);
		
		int[] erg = new int[moves.size()];
		for (int i = 0; i < moves.size(); i++) erg[i] = moves.get(i);
		return erg;
		
	}
	
	public int[] getAvailableMovesBinary() {
		int[] erg = board.clone();
		for (int i = 0; i < erg.length; i++) erg[i] = erg[i]==0?1:0;
		return erg;
	}
	
	public int getNext() {return this.next;}
	
	/**
	 * 
	 * @param x: zero based
	 * @param y: zero based
	 */
	public boolean move(int x, int y) {return this.move(x+3*y);}
	
	/**
	 * @param idx: zero based
	 */
	public boolean move(int idx) {
		if (idx >= 0 && idx < board.length && this.isWin()==0 && board[idx]==0) {
			board[idx]=next;
			next*=-1;
			return true;
		}
		return false;
	}
	
	public int isWin() {
		for (int i = 0; i < WINS.length; i++) {
			int sum = 0;
			for (int j = 0; j < WINS[i].length; j++) sum += board[WINS[i][j]];
			if (sum==3) return 1;
			if (sum==-3) return -1;
		}
		return 0;
	}
	
	public boolean isDraw() {
		if (this.isWin()!=0) return false;
		for (int i = 0; i < board.length; i++) if (board[i]==0) return false;
		return true;
	}
	
	public boolean isFinal() {return this.isDraw() || this.isWin()!=0;}
	
	
	@Override
	public String toString() {
		return 	"+-+-+-+\n" +
				"|" + getSymbol(board[0]) + "|" + getSymbol(board[1]) + "|" + getSymbol(board[2]) + "|\n" +
				"+-+-+-+\n" +
				"|" + getSymbol(board[3]) + "|" + getSymbol(board[4]) + "|" + getSymbol(board[5]) + "|\n" +
				"+-+-+-+\n" +
				"|" + getSymbol(board[6]) + "|" + getSymbol(board[7]) + "|" + getSymbol(board[8]) + "|\n" +
				"+-+-+-+";
	}
	
	public static String getSymbol(int i) {
		if (i==-1) return "O";
		if (i== 1) return "X";
		if (i== 0) return " ";
		return "#";
	}
	
	public static void check(int[] moves) {
		
		TicTacToe ttt = new TicTacToe();
		
    	System.out.println(ttt.toString());
    	for (int i = 0; i < moves.length; i++) {
    		
	    	
	    	System.out.println("------------------------------------------------------");
	    	System.out.print("avail : " + (ttt.getAvailableMoves().length!=0? "" : "none")); 
	    	for (int j = 0; j < ttt.getAvailableMoves().length; j++) System.out.print(ttt.getAvailableMoves()[j]);
			System.out.println();		
	    	System.out.println("move " + moves[i] + ": " + ttt.move(moves[i]));
	    	System.out.println(ttt.toString());
	    	System.out.println("result: " + (ttt.isWin()!=0? "win for '" + getSymbol(ttt.isWin()) + "'" : (ttt.isDraw()? "draw" : "none so far")));
    	}
	}
	
}
