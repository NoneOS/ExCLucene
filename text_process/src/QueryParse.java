import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.Jsoup;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;

import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.StandardCharsets;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

public class QueryParse {
	
	static final Pattern p_html = Pattern
			.compile("(http://|https://|ftp://|www)[A-Za-z_0-9\\./=@\\+\\-\\;]*");  // in this program, we don't filter 
	static final Pattern p_stop = Pattern.compile("[.]");
	static final Pattern p_cat = Pattern.compile("[-]");
	static final Pattern p_blank = Pattern.compile("\\s+");
	
	static int termMaxLength = 0;
	static String maxLength_term = "";
    public static void main(String[] args) throws IOException
    {
        
   	   String path = "/AOLquery/"; 
      String input_filename = path + "AOL.query.txt";
      String output_filename = path + "AOL.query.parse.txt";
        
        
        process_oneFile(input_filename, output_filename);
        System.out.println("maxLength_term: " + maxLength_term);
        System.out.println("termMaxLength: " + termMaxLength);
    }
    
    public static void process_oneFile(String input_filename, String output_filename)
    			throws IOException
    {
       FileReader fileReader = new FileReader(new File(input_filename));
       FileWriter fileWriter = new FileWriter(new File(output_filename));
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line;
        while( (line = bufferedReader.readLine()) != null )
        {
            String newline = filterLineString(line);
            fileWriter.append(newline + "\n");
        }
        fileWriter.flush();
        fileWriter.close();
        bufferedReader.close();

    }

	public static String filterLineString(String line)
	{
		
		String tempStr = line.trim();
		// filter html
//		Matcher m_html = p_html.matcher(tempStr);
//		int last_start = 0;
//		String newline = "";
//		while (m_html.find()) {
//			newline += tempStr.substring(last_start,
//					m_html.start());
//			last_start = m_html.end();
//		}
//		newline += tempStr.substring(last_start,
//				tempStr.length());
//		tempStr = newline;
		// System.out.println("tempStr(filter html):" + "[" +
		// tempStr + "]");
		
		tempStr = tempStr.replaceAll("[^\\p{ASCII}]", ""); // non-ASCII
		tempStr = tempStr.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
		tempStr = tempStr.replaceAll("[-]{1,}", " ");
		tempStr = tempStr.replaceAll("[.]{1,}", " ");
		tempStr = tempStr.replace("\0", " ");
		
		String ret = "";	
		//tempStr = tempStr.trim() + "\n";
		String[] rowStrArr = tempStr.trim().split("\\s+");
		for (int j = 0; j < rowStrArr.length; j++) {
			String curStr = rowStrArr[j];
			int curStrLength = curStr.length();
			if (curStrLength == 0 || curStrLength > 100)
			{	
//				System.out.print("in line, curStrLength == 0 || curStrLength > 300");
//				System.exit(1);
				continue;
			}
			if(curStrLength > termMaxLength)
			{
				maxLength_term = curStr;
				termMaxLength = curStrLength;
			}
//			Matcher m_stop = p_stop.matcher(curStr);
//			if (m_stop.find()) {
//				if (m_stop.end() == curStr.length()) {
//					curStr = curStr.substring(0,
//							curStr.length() - 1);
//				}
//				else
//					if(m_stop.end() == 1)
//						curStr = curStr.substring(1, curStr.length());
//			}
			
//			Matcher m_cat = p_cat.matcher(curStr);
//			if (m_cat.find()) {
//				if (m_cat.end() == curStr.length())
//					curStr = curStr.substring(0,
//							curStr.length() - 1);
//				else if (m_cat.end() == 1)
//					curStr = curStr.substring(1,
//							curStr.length());
//			}
			//if(curStr.isEmpty())
			//	continue;
			ret += curStr + " ";
		}
	//	ret = filterBlankSpace(ret);	
		return ret.trim();
	}
	
	public static String filterBlankSpace(String originalStr) {
		return originalStr.trim().replaceAll("\\s+", " ");
	}

}
