package de.hbo.dl4j.ttt.classifier;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.hbo.tictactoe.TicTacToe;


public class TTTGametreeGenerator {

    private static Logger log = LoggerFactory.getLogger(TTTGametreeGenerator.class);

    public static void main(String[] args) throws Exception {
    	
    	//TicTacToe.check(new int[] {0,0,4,8,1,6,7,4});
    	//TicTacToe.check(new int[] {0,1,2,3,5,4,6,8,7});
    	//TicTacToe.check(new int[] {0,1,2,0,0,0,0,0,0});
    	
    	writeGameTreeToFile(new File ("/home/august/workspace/projects/dl4j-testrange/src/main/resources/classification/tictactoe.txt"));
    	
    	List<TreeNode> tree = getGameTree();
    	System.out.println("size: " + tree.size());
    	
    	
    	//tree.forEach(System.out::println);
    	tree.forEach(each -> {
    		TicTacToe ttt = new TicTacToe(each.board);
    		//if (!ttt.isFinal()&&each.value!=0&&each.last==each.value) System.out.println(each);
    		//if (each.toString().startsWith("XOOX")) System.out.println(each);
    		//if (each.toString().startsWith("         ")) System.out.println(each);
    		//if (each.toString().substring(11, 13).equals(" |")) System.out.println(each);
    	});

    	int i = 0;
    	boolean done = false;
    	while (!done && ++i < 10) for (TreeNode each : tree) if (each.getLevel()==i && !each.toString().substring(12, 14).equals(" |")) {
    		System.out.println(each);
    		done = true;
    	}

    }
	
	public static List<TreeNode> getGameTree() {
		
		// build full tree
		List<TreeNode> tree = getGameTree(null);
		
		
		// sort by level
		List<List<TreeNode>> levels = new ArrayList<>();
		for (int i = 0; i < 10; i++) levels.add(new ArrayList<TreeNode>());
		tree.forEach(each -> levels.get(each.getLevel()).add(each));
		
		
		// condense levels
		levels.forEach(level -> {
			int vorher = level.size();
			for (int i = 0; i < level.size(); i++) for(int j = level.size()-1; j > i; j--) if (level.get(i).equals(level.get(j))) level.remove(j);
			System.out.println("level " + levels.indexOf(level) + ": " + vorher + " -> " + level.size());
		});
		
		
		// add parents/children
		for (int i = levels.size()-1; i > 0; i--) {
			
			List<TreeNode> parentLevel = levels.get(i-1);
			
			for (TreeNode each : levels.get(i)) for (int j = 0; j < each.getBoard().length; j++) if (each.getBoard()[j]==each.getLast()) {
				
				int[] board = each.getBoard().clone();
				board[j] = 0;
				
				for (TreeNode parent : parentLevel) if (parent.equals(new TreeNode(board))) {
					each.getParents().put(j, parent);
					parent.getChildren().put(j, each);
					break;
				}
			}
		};
		
		// derive value/labels
		for (int i = levels.size()-1; i >= 0; i--) { 
			
			for (TreeNode each : levels.get(i)) {
				
				TicTacToe ttt = new TicTacToe(each.getBoard());
				
				if (ttt.isFinal()) each.setValue(ttt.isWin());
				else {
    				int value = -1;
    				for (Integer key : each.getChildren().keySet()) {
    					TreeNode child = each.getChildren().get(key);
    					if (child.getLast()*child.getValue() > value) value=child.getLast()*child.getValue();
    				}
    				each.setValue(-value*each.getLast());
				}
			}
		};
		
		List<TreeNode> condensedTree = new ArrayList<>();
		levels.forEach(each->condensedTree.addAll(each));
		return condensedTree;
    	
	}
	
	private static List<TreeNode> getGameTree(TreeNode node) {
		
		if (node == null) node = new TreeNode(new TicTacToe().getBoard());
		
		List<TreeNode> erg = new ArrayList<>();
		erg.add(node);
		
		
		int[] moves = new TicTacToe(node.getBoard()).getAvailableMoves();
		for (int i = 0; i < moves.length; i++) {
			TicTacToe ttt = new TicTacToe(node.getBoard());
			ttt.move(moves[i]);
			erg.addAll(getGameTree(new TreeNode(ttt.getBoard())));
		}
		
		return erg;
    	
	}

	
	public static void writeGameTreeToFile(File file) throws IOException {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
			List<TreeNode> tree = getGameTree();
			for (int i = 0; i < tree.size(); i++) {
				if (i > 0) writer.newLine();
				writer.write(tree.get(i).toLabeledFileString()); 	
			}
		}
	}
    
    private static class TreeNode {

    	private final int[] board;
    	
    	private int value;
    	private final int last;
    	private final Map<Integer, TreeNode> parents = new HashMap<>();
    	private final Map<Integer, TreeNode> children = new HashMap<>();
    	
    	
    	public TreeNode(int[] input) {
    		
    		this.board = input.clone();

    		int num = 0;
    		for (int i = 0; i < this.board.length; i++) num+=Math.abs(this.board[i]);
    		this.last = num==0? 0 : (num%2==0? -TicTacToe.START:TicTacToe.START);
		}
    	
    	public int[] getBoard() {return this.board.clone();}
    	public int getLevel() {
    		int count = 0;
    		for (int i = 0; i < this.board.length; i++) if (this.board[i]==0) count++;
    		return this.board.length-count;
    	}
    	
    	public int getLast() {return this.last;}
    	public Map<Integer, TreeNode> getParents() {return this.parents;}
    	public Map<Integer, TreeNode> getChildren() {return this.children;}
    	
    	public void setValue(int value) {this.value=value;}
    	public int getValue() {return this.value;}
    	
    	public String getLabel() {
    		String erg = "";
    		for (int i = 0; i < this.board.length; i++) erg += ((this.children.get(i)!=null && this.getChildren().get(i).getValue()==this.value)? "1" : "0");
    		return erg;
		}
    	
    	public boolean equals(TreeNode other) {
    		if (other==null || other.getBoard()==null || other.getBoard().length!=this.board.length) return false;
    		for (int i = 0; i < this.board.length; i++) if (this.board[i]!=other.getBoard()[i]) return false;
    		return true;
		}
    	
    	public String toString() {
    		String erg = this.toString(this)+"|";
    		//for (TreeNode each : this.parents) erg+=this.toString(each.getBoard())+"|";
    		for (Integer key : this.children.keySet()) erg+=this.toString(this.children.get(key))+"|";
    		return erg;
    	}
    	
    	private String toString(TreeNode node) {
    		String erg = "";
    		for (int i = 0; i < node.getBoard().length; i++) erg += TicTacToe.getSymbol(node.getBoard()[i]);
    		erg+="-"+TicTacToe.getSymbol(node.getLast())+":"+TicTacToe.getSymbol(node.getValue());
    		return erg;
    	}
    	
    	public String toFileString() {
    		String erg = "";
    		for (int i = 0; i < this.getBoard().length; i++) erg += this.getBoard()[i] +", ";
    		erg+=this.getLast()+"|"+this.getValue();
    		return erg;
    	}
    	
    	public String toLabeledFileString() {
    		String erg = "";
    		for (int i = 0; i < this.getBoard().length; i++) erg += this.getBoard()[i] +", ";
    		erg+=this.getLast()+"|"+this.getLabel();
    		return erg;
    	}
    	
    }

}
































