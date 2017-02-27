import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.*;

public class URLParse {

	static final String folder_path = "./";
	
	static final String input_filename =  folder_path + "00.warc";
	static final String output_filename =  folder_path + "00.warc";

	static final Pattern p_stop = Pattern.compile("[.]");
	static final Pattern p_cat = Pattern.compile("[-]");
	static final Pattern p_blank = Pattern.compile("\\s+");
	
	public static void main(String[] args) throws IOException {
		System.out.println("start filterFile");
		filterFile(input_filename, output_filename);
	}
	
	public static void filterFile(String input_filename, String output_filename) throws IOException
	{
		File in_file = new File(input_filename);
		File out_file = new File(output_filename);
		
		StringBuffer sb = new StringBuffer();
		try{
			BufferedReader br = new BufferedReader(new FileReader(in_file)); 

			String line = null;
			int lineno = 0;
			while((line = br.readLine()) != null)
			{
				lineno++;
				assert(line.startsWith("clueweb09"));
				String ID = line;
				sb.append(ID + "\n"); // ID
				
				line = br.readLine();
				lineno++;
				
				String URL = filterString(line);
				if(URL.isEmpty())
				{
					System.out.println(ID);
				}
				sb.append(URL + "\n"); // URL
				
				line = br.readLine();
				lineno++;
				sb.append(line + "\n"); // content_lines
				
				int content_lines= Integer.parseInt(line);
				lineno += content_lines;
				int i = 0; 
				while(i < content_lines)
				{
					++i;
					line = br.readLine();
					sb.append(line + "\n"); // content line
				}
				line = br.readLine();
				sb.append(line + "\n"); // seperator line
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		if(deleteFile(input_filename))
			System.out.println("success delete " + input_filename);
		else
			System.out.println("fail delete " + input_filename);
		
		try
		{
			FileWriter fw_file = new FileWriter(out_file);
			fw_file.write(sb.toString());
			fw_file.flush();
			fw_file.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
    public static boolean deleteFile(String sPath) {
        boolean flag = false;
        File file = new File(sPath);
        if (file.isFile() && file.exists()) {
            file.delete();
            flag = true;
        }
        return flag;
    }
	
	public static String filterString(String line)
	{
		String tempStr = line;
		tempStr = tempStr.replaceAll("[^\\p{ASCII}]", ""); // non-ASCII
		tempStr = tempStr.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
		tempStr = tempStr.replaceAll("[-]{2,}", " ");
		tempStr = tempStr.replaceAll("[.]{2,}", " ");
		
		String ret = "";	
		tempStr = tempStr.trim() + "\n";
		String[] rowStrArr = tempStr.split("\\s+");
		for (int j = 0; j < rowStrArr.length; j++) {
			String curStr = rowStrArr[j];
			int curStrLength = curStr.length();
			if (curStrLength == 0 || curStrLength > 300)
			{
//				System.out.println("ID_str : " + ID_str);
//				System.out.println("rowStrArr.length: " + rowStrArr.length);
//				System.out.println("curStrLength: " + curStrLength);
//				System.out.println("curStr: " + curStr);	
				continue;
			}
			Matcher m_stop = p_stop.matcher(curStr);
			if (m_stop.find()) {
				if (m_stop.end() == curStr.length()) {
					curStr = curStr.substring(0,
							curStr.length() - 1);
				}
			}
			Matcher m_cat = p_cat.matcher(curStr);
			if (m_cat.find()) {
				if (m_cat.end() == curStr.length())
					curStr = curStr.substring(0,
							curStr.length() - 1);
				else if (m_cat.end() == 1)
					curStr = curStr.substring(1,
							curStr.length() - 1);
			}
			if(curStr.isEmpty())
				continue;
			ret += curStr + " ";
		}
		ret = filterBlankSpace(ret.trim());
		
		return ret;
	}
	
	public static String filterBlankSpace(String originalStr) {
		String tempStr = originalStr.trim();
		Matcher tempStr_m = p_blank.matcher(tempStr);
		if (tempStr_m.find())
			if (tempStr_m.end() == tempStr.length()) {
				tempStr += "\n";
				tempStr = tempStr.replaceAll("\\s*\n", "");
				// System.out.println("tempStr:" + "[" + tempStr + "]");
			} else {
				tempStr = tempStr.replaceAll("\\s+", " ");
				tempStr += "\n";
				tempStr = tempStr.replaceAll("\\s*\n", "");

			}
		return tempStr;
	}
	
}
