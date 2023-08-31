function img = loadImage(ds, frame_idx)
    kitti_path = 'datasets/kitti';
    malaga_path = 'datasets/malaga-urban';
    images = dir([malaga_path '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    parking_path = 'datasets/parking';
    % get next frame
    if ds == 0
        img = imread([kitti_path '/05/image_0/' sprintf('%06d.png', frame_idx)]);
    elseif ds == 1
        img = rgb2gray(imread([malaga_path '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' left_images(frame_idx).name]));
    elseif ds == 2
        img = im2uint8(rgb2gray(imread([parking_path sprintf('/images/img_%05d.png', frame_idx)])));
    else
        assert(false);
    end
end