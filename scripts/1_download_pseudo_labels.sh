echo 'PLEASE FILL OUT SURVEY FOR INIT LABEL DOWNLOAD LINK!'
# curl -o ckpts.zip 'None'
mkdir save/example
mkdir save/example/L0_pseudo_labels
mv init_pseudo_labels.zip save/example/L0_pseudo_labels/
cd save/example/L0_pseudo_labels
unzip init_pseudo_labels.zip
rm init_pseudo_labels.zip
cd ../../..
