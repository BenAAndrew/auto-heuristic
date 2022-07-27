function predict(category) {
  // Accuracy: 100%
  if (category == "c") {
    return "virginica";
  } else {
    if (category == "a") {
      return "setosa";
    } else {
      return "versicolor";
    }
  }
}
