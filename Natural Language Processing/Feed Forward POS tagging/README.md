# Feed-Forward POS tagger using keras

Note that input words are represented with a 100 dimensional embedding, and the network has one hidden layer with 100 nodes.




&nbsp;
&nbsp;
&nbsp;

************ Output ************


Each model iterates 3 (epochs = 3) times to compute accuracy of test dataset.



First model is complied with binary_crossentropy as a loss and sgd as an optimizer. 



Second model is compiled with binary_crossentropy as a loss and adam as an optimizer. I added Activation('softmax') to output layer.



Third model is compiled with mean_square_error as a loss and adam as an optimizer.



Last model is compiled with mean_square_error as a loss and sgd as an optimizer.

I attached the testing result below:



Test loss and accuracy with loss: binary_crossentropy, optimizer: sgd

[0.1583254771602528, 0.97777706774416351]



Test loss and accuracy with loss: binary_crossentropy, optimizer: adam

[0.085717181083884972, 0.97777706774416351]



Test loss and accuracy with loss: mean_squared_error, optimizer: adam

[0.0021856717593628284, 0.92675999642719431]



Test loss and accuracy with loss: mean_squared_error, optimizer: sgd

[0.10288566864992961, 0.053694427049387143]




I found that if I add "add(Activation('softmax'))" to model, then the train accuracy increases very quickly. For example, model 2's accuracy starts 0.97. But other models' train accuracies increase smoothly.



As seen above, binary_crossentropy loss produces great results. Combination of "loss: mean_squared_error and optimizer: adam" returns pretty good result, but not as good as models' with binary_crossentropy loss. The last model gives pretty bad accuracy. 



