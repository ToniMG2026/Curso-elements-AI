# CabinVision: AI-Powered Price Estimator
Building AI course project

## Summary
CabinVision uses neural networks to provide instant price estimates for holiday cabins based on size, location, and amenities. It aims to simplify the valuation process for owners and buyers alike, ensuring fair market pricing. 

## Background
Valuing remote properties is often inconsistent and requires expensive manual appraisals. This project solves that by:
* Providing instant, data-driven price benchmarks.
* Reducing bias in property negotiations.
* Helping small-scale investors identify undervalued properties.

My motivation stems from a personal interest in real estate transparency and the technical challenge of applying deep learning to small, specific datasets.

## How is it used?
The solution is designed for property owners and real estate agents. A user enters cabin attributes (square footage, number of rooms, distance to water) into a simple interface. The model then performs a forward pass to return a predicted price.

![Cabin Architecture](https://media.istockphoto.com/id/2084953046/es/foto/nodos-de-redes-neuronales-aprendizaje-profundo-inteligencia-artificial-modelo-de-aprendizaje.jpg?s=2048x2048&w=is&k=20&c=YSnr7tSqXFuEf54cweHn98dVYKtSpn3SFROm0ClI4II=)

Example of the prediction logic:
```python
def predict_price(features, weights, bias):
    # Perform the forward pass
    z = np.dot(features, weights) + bias
    return np.maximum(0, z) # ReLU activation

Data sources and AI methodsThe data is sourced from public real estate listings and historical sales records.Method: A Multi-Layer Perceptron (MLP) neural network.Activations: ReLU for hidden layers to capture non-linear relationships and Linear activation for the output layer.LayerTypeActivationInput5 NodesN/AHidden2 Dense LayersReLUOutput1 NodeLinearChallengesThe project does not account for subjective "charm" or the specific condition of interior finishes. Ethical considerations include the risk of automated systems reinforcing historical pricing biases or being used by large corporations to out-compete local buyers.What next?The next step is integrating Computer Vision to analyze photos of the cabins, allowing the AI to "see" the quality of the view or the modernness of the kitchen. I would need to improve my skills in Convolutional Neural Networks (CNNs) to achieve this.AcknowledgmentsInspired by the Elements of AI curriculum.Logic structures based on the Building AI course exercises.Sleeping Cat on Her Back by Umberto Salvagnin / CC BY 2.0
---

### A few tips for your GitHub setup:

* **The Pen Icon:** When you're in your repository on GitHub, click the **README.md** file, then click the small pencil icon in the top right to enter "Edit" mode.
* **Commit Changes:** After pasting the code, scroll to the bottom, write a short message like "Initial project plan draft," and click **Commit changes**.
* **Images:** If you have a specific diagram you made, upload it to your GitHub repo first. Then, change the image link in the markdown from the `https://...` URL to just the filename (e.g., `![My Diagram](my_diagram.png)`).

How does this look for your project idea? If you have a different topic in mind (like a movie recommend
