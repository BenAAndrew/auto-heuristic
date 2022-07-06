function predict(petalWidth, petalLength) {
  if (petalLength <= 2.45) {
    return "setosa";
  } else {
    if (petalLength <= 4.75) {
      if (petalWidth <= 1.65) {
        return "versicolor";
      } else {
        return "virginica";
      }
    } else {
      if (petalWidth <= 1.75) {
        return "versicolor";
      } else {
        return "virginica";
      }
    }
  }
}
