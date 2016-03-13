package debateCrawler;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;

import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.TwitterObjectFactory;

public class SaveTweets
{
    /**
     * Replace newlines and tabs in text with escaped versions to making printing cleaner
     *
     * @param text
     *            The text of a tweet, sometimes with embedded newlines and tabs
     * @return The text passed in, but with the newlines and tabs replaced
     */
    public static String cleanText(String text)
    {
        text = text.replace("\n", "\\n");
        text = text.replace("\t", "\\t");

        return text;
    }

    // Storing function
    // inputs:
    // targetFile:
    // Result: query result to be store at this time.
    // apendToFile: true if the input tweets should be appended to the input file contents.
    // false if hte input tweets should be written to the file, overwriting any existing contents.
    public static void storeQueryResult(String targetFile, List<Status> tweets, String header, boolean appendToFile) throws Exception
    {
        FileOutputStream fos = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        boolean firstWrite = true;

        if (!appendToFile)
        {
            System.out.println("Writing " + tweets.size() + " tweets to file: " + targetFile);
        } else
        {
            System.out.println("Appending " + tweets.size() + " tweets to file: " + targetFile);

        }

        try
        {
            // open file with append file set to as specified
            fos = new FileOutputStream(targetFile, appendToFile);
            osw = new OutputStreamWriter(fos, "UTF-8");
            bw = new BufferedWriter(osw);
            for (Status status : tweets)
            {

                if ((firstWrite) && (!appendToFile))
                {
                    // Start a new file = write a header, removing any existing contents.
                    bw.write(header);
                    bw.newLine();
                    firstWrite = false;
                }

                String rawJSON = cleanText(TwitterObjectFactory.getRawJSON(status));

                bw.write(rawJSON);
                bw.newLine();
                bw.flush();
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.out.println("Failed to store tweets: " + e.getMessage());
            throw e;
        }
        finally
        {
            if (bw != null)
            {
                try
                {
                    bw.close();
                }
                catch (IOException ignore)
                {
                    System.out.println("Ignoring exception when closing writer: " + ignore.getMessage());
                }
            }
            if (osw != null)
            {
                try
                {
                    osw.close();
                }
                catch (IOException ignore)
                {
                    System.out.println("Ignoring exception when closing writer: " + ignore.getMessage());
                }
            }
            if (fos != null)
            {
                try
                {
                    fos.close();
                }
                catch (IOException ignore)
                {
                    System.out.println("Ignoring exception when closing writer: " + ignore.getMessage());
                }
            }
        }
    }
}