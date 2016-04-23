package ConvertToCSV;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class cleanCsv
{

    // Usage:
    // C:\Users\doylerav\Documents\GitHub\data-mining-debate-analysis\r\excel\filtered\march10th_before_aggAndFilt.csv.txt
    // C:\Users\doylerav\Documents\GitHub\data-mining-debate-analysis\r\excel\filtered\march10th_before_aggAndFiltforR.csv
    // 20000
    // C:\Users\doylerav\Documents\GitHub\data-mining-debate-analysis\r\excel\filtered\march10th_after_aggAndFilt.csv.txt
    // C:\Users\doylerav\Documents\GitHub\data-mining-debate-analysis\r\excel\filtered\march10th_after_aggAndFiltforR.csv
    // 30000

    // Cleans a csv file of line breaks in the middle of tweet text
    // (these cause ingestion issues)
    public static void main(String[] args) throws Exception
    {
        if (args.length < 2)
        {
            System.out.println("CleanCsv <input csv> <output csv> <id offset #>");
            return;
        }

        int idOffset = 0;
        if (args.length > 2)
        {
            idOffset = Integer.parseInt(args[2]);
        }

        File src = new File(args[0]);
        File dest = new File(args[1]);
        String absolutePath = src.getAbsolutePath();

        if (absolutePath.equalsIgnoreCase(dest.getAbsolutePath()))
        {
            System.err.println("Source cannot be the same as destination!");
            return;
        }

        System.out.println("Reading file at " + absolutePath);
        System.out.println("Reading file at " + absolutePath);

        CSVReader reader = null;
        CSVWriter writer = null;

        try
        {
            String[] headerFields = null;

            // First line is a header
            reader = new CSVReader(new FileReader(src.getAbsolutePath()));
            writer = new CSVWriter(new FileWriter(dest, false));

            headerFields = reader.readNext();

            if (headerFields == null || headerFields.length <= 0)
            {
                throw new Exception("Could not read header line from csv");
            } else
            {
                System.out.println("Ingesting " + args[0]);
            }

            // Add a id field to the csv.
            String[] newHeader = new String[headerFields.length];

            int textIndex = -1;
            int tweetIdIndex = 1;
            int tweetLocationIndex = -1;
            int tweetUserTimezoneIndex = -1;
            int idIndex = -1;
            int candidateIndex = -1;
            boolean idFound = false;

            for (int i = 0; i < headerFields.length; i++)
            {
                newHeader[i] = headerFields[i].trim();

                if ((newHeader[i].length() > 0) && ((int) newHeader[i].charAt(0) == 65279))
                {
                    newHeader[i] = newHeader[i].substring(1);
                }

                if (newHeader[i].equalsIgnoreCase("id"))
                {
                    idFound = true;
                    idIndex = i;
                    continue;
                }

                if (newHeader[i].equalsIgnoreCase("text"))
                {
                    textIndex = i;
                    continue;
                }

                if (newHeader[i].equalsIgnoreCase("candidate") || newHeader[i].equalsIgnoreCase("candiate"))
                {
                    candidateIndex = i;
                    continue;
                }

                if (newHeader[i].equalsIgnoreCase("tweet_id"))
                {
                    tweetIdIndex = i;
                    continue;
                }

                if (newHeader[i].equalsIgnoreCase("tweet_location"))
                {
                    tweetLocationIndex = i;
                    continue;
                }

                if (newHeader[i].equalsIgnoreCase("user_timezone"))
                {
                    tweetUserTimezoneIndex = i;
                    continue;
                }

            }

            // Get the expected input field length (before we insert an id field, which the input
            // won't have)
            int expectedFieldLength = headerFields.length;

            // if we need to insert an id field, use the new header we built.
            if (!idFound)
            {
                // Add it on the end so we don't have to worry about shifting other columns.

                String[] headerWithId = new String[newHeader.length + 1];
                for (int i = 0; i < newHeader.length; i++)
                {
                    headerWithId[i] = newHeader[i];
                }

                headerWithId[newHeader.length] = "id";
                idIndex = newHeader.length;
                newHeader = headerWithId;

            }

            headerFields = newHeader;

            if (textIndex == -1)
            {
                throw new Exception("No text column found!");
            }

            writer.writeNext(headerFields);

            int id = idOffset;
            String[] fields = null;
            while ((fields = reader.readNext()) != null)
            {
                id++;

                if ((expectedFieldLength != fields.length) || fields[0].isEmpty())
                {
                    System.err.println("Header fields must match # of fields; first field can't be empty!");
                    System.err.println(fields[0]);

                    continue;
                }

                // the header got inserted a couple times in the
                // json to csv code. If the first field value is tweet_id, skip this row.
                if (fields[0].equalsIgnoreCase("tweet_id") || fields[1].equalsIgnoreCase("tweet_id")
                        || fields[2].equalsIgnoreCase("tweet_id"))
                {
                    continue;
                }

                // A couple fields have the string "null" which R doesn't like.
                if (fields[tweetLocationIndex].equalsIgnoreCase("null"))
                {
                    fields[tweetLocationIndex] = "none";
                }
                if (fields[tweetUserTimezoneIndex].equalsIgnoreCase("null"))
                {
                    fields[tweetUserTimezoneIndex] = "none";
                }

                switch (fields[candidateIndex])
                {
                case "Trump":
                case "Donald Trump":
                    fields[candidateIndex] = "Donald Trump";
                    break;
                case "Rubio":
                case "Marco Rubio":
                    fields[candidateIndex] = "Marco Rubio";
                    break;
                case "Cruz":
                case "Ted Cruz":
                    fields[candidateIndex] = "Ted Cruz";
                    break;
                case "Kasich":
                case "John Kasich":
                    fields[candidateIndex] = "John Kasich";
                    break;
                default:
                    fields[candidateIndex] = "Unknown";
                    break;
                }

                if (!idFound)
                {
                    // Add the id.
                    String[] newFields = new String[fields.length + 1];

                    // Copy existing fields and convert a couple fields that
                    // tend to be in the original filtered files.
                    for (int i = 0; i < fields.length; i++)
                    {
                        newFields[i] = fields[i];
                        if (fields[i].equalsIgnoreCase("null"))
                        {
                            newFields[i] = "none"; // R has special processing for null we want to
                                                   // avoid.
                        }
                        if (fields[i].equals("neutral"))
                        {
                            newFields[i] = "Neutral"; // R is case sensitive, collapse to the Aug
                                                      // value
                        }
                        if (fields[i].equals("negative"))
                        {
                            newFields[i] = "Negative"; // R has special processing for null we want
                                                       // to avoid.
                        }
                        if (fields[i].equals("positive"))
                        {
                            newFields[i] = "Positive"; // R has special processing for null we want
                                                       // to avoid.
                        }
                    }

                    // Add our id.
                    newFields[idIndex] = Integer.toString(id);

                    fields = newFields;
                }

                // A couple cleanups on the text column so we can ingest as a csv in r or excel
                String text = new String(fields[textIndex]);

                // This is an attempt to catch some columns that split because of a comma and no
                // paired quotes.
                if (fields.length > textIndex + 1 && headerFields[textIndex + 1] == "")
                {
                    text = text + ", " + fields[textIndex + 1];
                }
                if (fields.length > textIndex + 2 && headerFields[textIndex + 2] == "")
                {
                    text = text + ", " + fields[textIndex + 2];
                }

                // A couple utf-8 representations of the "this tweet was truncated" character that
                // might sneak in
                // if (text.matches("[^\\x00-\\x7F]"))
                if (!isPureAscii(text))
                {
                    System.out.println("Non ascii in text " + text);
                }

                String newText = "";

                for (int i = 0; i < text.length(); i++)
                {
                    char c = text.charAt(i);

                    if ((c == 0xFFFD) && i >= 138)
                    {
                        // strip these characters at end of line
                        continue;
                    }

                    if ((c == '\n') || (c == '\r'))
                    {
                        c = ' ';
                    }

                    if (c >= 0x80)
                    {
                        continue;
                    }

                    newText = newText + c;
                }

                // R output sometimes adds this for the '...' bit
                newText.replace("<U+FFFD>", " ");

                text = newText;
                if (!isPureAscii(fields[textIndex]))
                {
                    System.out.println("New text = " + text);
                    System.out.println("tweet id " + fields[1]);
                }

                fields[textIndex] = text;

                writer.writeNext(fields);
            }

            System.out.println("Done with ingestion of " + args[0] + ", output written to " + args[1]);

        }
        catch (

        Exception e)

        {
            e.printStackTrace();
            throw e;
        }
        finally

        {
            if (reader != null)
            {
                reader.close();
            }

            if (writer != null)
            {
                writer.close();
            }
        }

    }

    static CharsetEncoder asciiEncoder = Charset.forName("US-ASCII").newEncoder(); // or
                                                                                   // "ISO-8859-1"
                                                                                   // for ISO Latin
                                                                                   // 1

    public static boolean isPureAscii(String v)
    {
        return asciiEncoder.canEncode(v);
    }

    public static String encodeToAscii(String v)
    {

        String newText = v;
        try
        {
            CharBuffer uCharBuffer = CharBuffer.wrap(v);

            // Encode the remaining content of a single input character buffer to a new byte buffer.
            // Converts to ISO-8859-1 bytes and stores them to the byte buffer

            ByteBuffer bbuf = asciiEncoder.encode(uCharBuffer);

            newText = bbuf.toString();
        }
        catch (CharacterCodingException e)
        {
            System.out.println("Character Coding Error: " + e.getMessage());
        }
        finally
        {
            asciiEncoder.reset();
        }
        return newText;

    }
}
