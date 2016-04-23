package ConvertToCSV;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import twitter4j.Status;
import twitter4j.TwitterException;
import twitter4j.TwitterObjectFactory;

public final class ConvertToCSV
{
    public static void main(String[] args)
    {
        if (args.length != 2)
        {
            System.out.println("Usage: ConvertToCSV JSON_source_dir output_csv");
            System.exit(-1);
        }
        try
        {
            File[] files = new File(args[0]).listFiles(new FilenameFilter() {
                // @Override
                public boolean accept(File dir, String name)
                {
                    return name.endsWith(".json");
                }
            });
            for (File file : files)
            {
                List<Status> restoredTweets = readLines(file);
                String candidate = getCandidateFromFileName(file.getName());
                saveToCSV(restoredTweets, args[1], candidate, false);
                System.out.println("saved " + file.getName() + " to " + args[1]);
            }
            System.exit(0);
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.out.println("Failed to store tweets: " + e.getMessage());
            System.exit(-1);
        }

    }

    public static String getCandidateFromFileName(String fileName)
    {
        if (fileName.indexOf("Before") != -1)
            return fileName.substring(0, fileName.indexOf("Before"));
        else if (fileName.indexOf("After") != -1)
            return fileName.substring(0, fileName.indexOf("After"));
        else
            return "no candidate found";
    }

    public static void saveToCSV(List<Status> tweets, String targetCSV, String candidate, boolean filterOutRetweet)
            throws Exception
    {
        CSVWriter writer = null;
        CSVReader reader = null;
        try
        {
            new File("savedTweets").mkdir();

            if (targetCSV.endsWith(".csv") || targetCSV.endsWith(".CSV"))
                targetCSV = targetCSV.substring(0, targetCSV.length() - 4);
            File targetCSVFile = new File(targetCSV + ".csv");

            Hashtable<String, String[]> lookupTable = new Hashtable<String, String[]>();

            if (targetCSVFile.exists())
            {
                reader = new CSVReader(new FileReader(targetCSVFile));

                String[] nextLine;
                while ((nextLine = reader.readNext()) != null)
                {
                    // nextLine[] is an array of values from the line
                    lookupTable.put(nextLine[0], nextLine);
                }
                reader.close();
                reader = null;

                writer = new CSVWriter(new FileWriter(targetCSVFile, true));
            } else
            {
                writer = new CSVWriter(new FileWriter(targetCSVFile, true));
                String[] headers = { "tweet_id", "candidate", "name", "tweet_created", "tweet_location",
                        "user_timezone", "retweet_count", "text" };
                writer.writeNext(headers);
            }

            SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy HH:mm");

            for (Status status : tweets)
            {
                // if retweet need to be filtered out.
                if ((filterOutRetweet) && status.isRetweet())
                    continue;
                // specify fields to save to CSV
                String[] tweetRecord = { String.valueOf(status.getId()), candidate, status.getUser().getName(),
                        formatter.format(status.getCreatedAt()), status.getUser().getLocation(),
                        status.getUser().getTimeZone(), String.valueOf(status.getRetweetCount()), status.getText() };

                if (lookupTable.get(tweetRecord[0]) == null)
                {
                    lookupTable.put(tweetRecord[0], tweetRecord);
                    writer.writeNext(tweetRecord);
                }
            }
        }
        finally
        {
            if (reader != null)
            {
                try
                {
                    reader.close();
                }
                catch (IOException ignore)
                {
                }
            }

            if (writer != null)
            {
                try
                {
                    writer.flush();
                    writer.close();
                }
                catch (IOException ignore)
                {
                }
            }
        }

    }

    private static List<Status> readLines(File fileName) throws Exception
    {
        FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null;
        List<Status> ret = new ArrayList<Status>();
        try
        {
            fis = new FileInputStream(fileName);
            isr = new InputStreamReader(fis, "UTF-8");
            br = new BufferedReader(isr);
            br.readLine();
            String nextLine = br.readLine();
            while (nextLine != null)
            {
                try
                {
                    ret.add(TwitterObjectFactory.createStatus(nextLine));
                }
                catch (TwitterException e)
                {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
                nextLine = br.readLine();
            }
            return ret;

        }
        finally
        {
            if (br != null)
            {
                try
                {
                    br.close();
                }
                catch (IOException ignore)
                {
                }
            }
            if (isr != null)
            {
                try
                {
                    isr.close();
                }
                catch (IOException ignore)
                {
                }
            }
            if (fis != null)
            {
                try
                {
                    fis.close();
                }
                catch (IOException ignore)
                {
                }
            }
        }
    }

}