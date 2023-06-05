import matplotlib.pyplot as plt
import numpy as np

def single_bar():
  # Create data
  categories = ['VGG19', 'EfficientNet']
  values = [0.839086, 0.863801]

  # Create a figure and axis
  fig, ax = plt.subplots()

  # Plot data
  ax.bar(categories, values)

  # Set title and labels
  ax.set_title('MesoNet Custom with Transfer Learning')
  ax.set_xlabel('Models')
  ax.set_ylabel('Acuracy')

  # Display the plot
  plt.show()
  plt.savefig("graphs/mesocustomaccvgg19effnet.png")

def double_bar():
  # Create data
  categories = ['VGG19', 'EfficientNet']
  tp = [681, 754]
  fp = [822, 792]
  tn = [95, 125]
  fn = [192, 119]

  # Get the number of categories
  N = len(categories)

  # Create an array with the position of each bar along the x-axis
  ind = np.arange(N) 

  # Figure size
  plt.figure(figsize=(10,5))

  # Width of each bar
  width = 0.1       

  # Plotting
  plt.bar(ind, tp, width, label='True Positives')
  plt.bar(ind + width, fp, width, label='False Positives')
  plt.bar(ind + width * 2, tn, width, label='True Negatives')
  plt.bar(ind + width * 3, fn, width, label='False Negatives')

  plt.ylabel('Quantity')
  plt.title('MesoNet Custom with Transfer Learning')

  # xticks()
  # First argument - A list of positions at which ticks should be placed
  # Second argument -  A list of labels to place at the given locations
  plt.xticks(ind + width * 1.5, categories)

  # Finding the best position for legends and putting it
  plt.legend(loc='best')
  plt.show()
  plt.savefig("graphs/mesocustomaccvgg19effnetfscore.png")

def models_acc():
    models = ['efficientnetb0', 'mesonet', 'mesonet_custom_512_with_efficientnet', 
          'vgg19', 'mesonet_custom', 'efficientnetb7', 'mesonet_custom_512', 
          'mesonet_custom_512_with_vgg19', 'efficientnetb7_custom', 'resnet_v2_50']

    means = [0.487870, 0.212585, 0.863801, 0.4972718, 0.710437, 
            0.487870, 0.732507, 0.839086, 0.555296, 0.487870]

    plt.figure(figsize=(25,6))

    plt.barh(models, means, color='skyblue')

    plt.xlabel('Accuracy')
    plt.title('Accuracy of the Models')

    plt.show()
    plt.savefig("graphs/allmodelsacc.png")


def main():
  single_bar()
  double_bar()
  models_acc()
  pass
if __name__ == '__main__':
    main()