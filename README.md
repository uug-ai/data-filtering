# Data filtering

Data filtering is an essential process in managing the vast amounts of video data generated by video cameras and CCTV systems. In environments where bandwidth and storage are limited, it is crucial to ensure that only relevant video recordings are forwarded to the cloud for further analysis and storage. This is where machine learning (ML) models become invaluable. By employing ML models, data filtering can be significantly enhanced to automatically assess and determine the importance of each video recording based on specific conditions or criteria. For example, an ML model can be trained to detect particular objects or events within the video footage, such as identifying the presence of at least five people or recognizing at least two trucks. When these predefined conditions are met, the recording is marked as relevant and subsequently forwarded to the cloud. This intelligent filtering mechanism not only optimizes the use of network resources but also ensures that critical video data is prioritized and available for real-time decision-making and analytics. Additionally, by reducing the volume of video data transmitted to the cloud, organizations can achieve cost savings on storage and processing while maintaining the integrity and relevance of their video streams. This approach enhances the efficiency and effectiveness of video surveillance systems, ensuring that only pertinent data is retained and analyzed.

![Data filtering with Kerberos Vault](./assets/images/data-filtering-with-kerberos-vault.png)

As illustrated in the image above, recordings are collected from a heterogeneous landscape through Kerberos Agents and stored centrally in a local Kerberos Vault. Each stored recording triggers an integration with the data filtering service. This data filtering can involve any type of analysis algorithm, such as computer vision or machine learning models, to identify specific events and mark them as important. Once a recording is marked as important, the data filtering service instructs the local Kerberos Vault to forward the recording to a remote Kerberos Vault. By implementing this process, we achieve the following benefits:

- Reduced bandwidth usage by minimizing the number of recordings being forwarded
- Decreased cloud storage requirements due to fewer recordings being transmitted
- Enhanced relevance of recordings, as most irrelevant recordings are not forwarded
- Automated creation of datasets.

## Datascience

One of the most important steps in data science or the development of machine learning models is the collection of data. Having high-quality data is the key ingredient to training any successful algorithm. Unfortunately, collecting data, especially qualitative data, is one of the most complicated steps. Imagine you want to train a model that recognizes people wearing purple sunglasses, a red shirt, and yellow shoes. Think about it, where would you be able to find a dataset that includes people meeting those criteria? It would not be an easy task, right?

In most cases, data scientists spend a significant amount of their time finding the correct data, and this is typically a very time-consuming task. Especially in the world of video, typically terabytes of data are analyzed or labeled manually, and most of the time the correct dataset is not readily available. To make it even worse, machine learning models require up-to-date training data, which means that collecting data is not a one-time job; it's a continuous process that never stops. This means that the developers of these models require an automated and smart approach for creating those datasets. This is where data filtering becomes crucial for the concept of automated dataset harvesting.

Automated dataset harvesting involves using algorithms and tools to continuously collect, filter, and update datasets to ensure they meet the required criteria for training machine learning models. This approach not only saves time but also ensures that the data remains relevant and high-quality, which is essential for the success of any machine learning project.

## What's in this project

This project includes a Python program, `queue_filter.py`, which leverages the YOLOv8 framework to analyze recordings stored in the local Kerberos Vault. Whenever a recording is stored, Kerberos Vault sends an event/message to the RabbitMQ message broker. The `queue_filter.py` script reads these events from RabbitMQ, downloads the relevant recordings from Kerberos Vault, and processes them using the YOLOv8 model via the `processFrame` function.

    condition = "4 persons detected"
    frame, total_time_class_prediction, conditionMet = processFrame(
        MODEL, frame, video_out, condition)

    if conditionMet:
        print(
            "Condition met, stopping the video loop, and forwarding video to remote vault")
        # @TODO: Forward the video to the remote vault.
        break

    # Increase the frame_number and predicted_frames by one.
    predicted_frames += 1

Upon evaluation, the `processFrame` function attempts to identify a match based on specified conditions, such as detecting "4 persons" or "10 cars." If the YOLOv8 model successfully detects, for instance, 4 persons or 10 cars, the function will return a positive match. Consequently, an API call will be made to request Kerberos Vault to forward the recording to the remote Kerberos Vault.
