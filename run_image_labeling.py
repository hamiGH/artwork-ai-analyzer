import time
import tensorflow as tf
from src import load_params, set_device_option, InferenceModel, PrepareData


if __name__ == "__main__":
    start_time = time.time()

    # load parameters
    params = load_params("config/config.yml")

    # set device option
    set_device_option(params['device'])

    with tf.device(params['device']):
        # initiate objects
        data_object = PrepareData(config_dir="config")
        model_object = InferenceModel(config_dir="config")

        # begin experiment
        if params['phase'] == "train":
            train_generator, validation_test_dataset, classes = data_object.create_train_val_data(
                model_object.batch_size)
            history, pred, labels = model_object.train_model(train_generator, validation_test_dataset, classes)
            model_object.plot_training_performance(history)
        else:
            test_images, _ = data_object.read_data()
            predictions = model_object.test_model(test_images)
            model_object.plot_test_images(test_images, data_object.classes)

        print("--- %s seconds ---" % (time.time() - start_time))
