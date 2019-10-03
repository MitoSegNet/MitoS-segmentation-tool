/*
 * 
 * Converts tif or non-tif images or stacks into max intensity projected 8-bit tif files for subsequent MitoSegNet
 * segmentation
 * 
 */

Dialog.create("Configure");
Dialog.addChoice("One or multiple folders?", newArray("One", "Multiple"));
Dialog.show();
 
nfolders = Dialog.getChoice();

in = getDirectory("select folder");
names_org = getFileList(in);

save_path = in + File.separator + "MaxInt_Tiff_Files";

if (File.isDirectory(save_path)){
	print("Directory already exists");
}

else{
	File.makeDirectory(in + File.separator + "MaxInt_Tiff_Files");
}


function convert(in, out, image){
	
	Ext.openImagePlus(in + File.separator + image);

	getDimensions(width, height, channels, slices, frames);
	
	if(slices>1){

		run("Z Project...", "projection=[Max Intensity]");
		selectWindow("MAX_" + image);
		
	}	
	run("8-bit");

	saveAs("Tiff", out + File.separator + image);
	run("Close All");

	run("Collect Garbage");
}

setBatchMode(true);
run("Bio-Formats Macro Extensions");

if(nfolders == "Multiple"){

	for(i=0; i<names_org.length; i++){
		
		out = in + File.separator + "MaxInt_Tiff_Files" + File.separator + names_org[i];
		File.makeDirectory(out);

		folder_path = in + File.separator + names_org[i];
		sub_names_org = getFileList(folder_path);

		for(n=0; n<sub_names_org.length; n++){

			convert(folder_path, out, sub_names_org[n]);			
		}		
	}
		
}
else{

	for(i=0; i<names_org.length; i++){

		convert(in, save_path, names_org[i]);	
	}
}
