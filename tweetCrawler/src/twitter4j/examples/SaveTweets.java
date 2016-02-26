package twitter4j.examples;

import twitter4j.*;

import java.io.*;
import java.util.List;

public class SaveTweets {
	private int maxCount;
	private String fileBase;
	private String currentFile;
	private int nextThreshold;
	
	public int getMaxCount() {
		return maxCount;
	}

	public void setMaxCount(int maxCount) {
		this.maxCount = maxCount;
	}

	public String getFileBase() {
		return fileBase;
	}

	public void setFileBase(String fileBase) {
		this.fileBase = fileBase;
	}

	public int getNextThreshold() {
		return nextThreshold;
	}

	public void setNextThreshold(int nextThreshold) {
		this.nextThreshold = nextThreshold;
	}

	public String getCurrentFile() {
		return currentFile;
	}

	public void setCurrentFile(String currentFile) {
		this.currentFile = currentFile;
	}

	SaveTweets(int maxCount, String fileBase) {
		this.maxCount = maxCount;
		this.fileBase = fileBase;
        this.currentFile = this.fileBase + "_0.json";
        this.nextThreshold = maxCount;
	}
	
	public void storeQueryResult(int currentCount, QueryResult Result) {
		FileOutputStream fos = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        try {
            fos = new FileOutputStream(currentFile,true);
            osw = new OutputStreamWriter(fos, "UTF-8");
            bw = new BufferedWriter(osw);
            for (Status status : Result.getTweets()) {
            	String rawJSON = TwitterObjectFactory.getRawJSON(status);
            	bw.write(rawJSON);
                bw.newLine();
                bw.flush();
           }    
        }catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to store tweets: " + e.getMessage());
        } finally {
        	if (Result.getCount() + currentCount >= nextThreshold) {
        		nextThreshold += maxCount;
        		this.currentFile = this.fileBase + "_" + String.valueOf(currentCount+Result.getCount())+".json";
        	}
            if (bw != null) {
                try {
                    bw.close();
                } catch (IOException ignore) {
                }
            }
            if (osw != null) {
                try {
                    osw.close();
                } catch (IOException ignore) {
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException ignore) {
                }
            }
        }
	}  
}