using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Features2D;
using System.Diagnostics;

namespace ShapeDetection
{
    public partial class Main : Form
    {
        //declaring global variables
        private Capture capture = null; //takes images from camera as image frames
        private bool captureInProgress; // checks if capture is executing
        Mat PreviousFrame = new Mat();
        private int State = 0;
        VectorOfPoint contours = new VectorOfPoint();
        Point[] points = new Point[] { new Point { X = 0 , Y = 0 }, new Point { X = 0 , Y = 0 }, new Point { X = 0, Y = 0 }, new Point { X = 0, Y = 0 }, new Point { X = 0, Y = 0 },
                         new Point { X = 0, Y = 0 }, new Point { X = 0, Y = 0 }, new Point { X = 0, Y = 0 }, new Point { X = 0, Y = 0 }, new Point { X = 0, Y = 0 } };
        Mat[] templ = new Mat[10];
        int framecount5 = 0;
        Point[] points5behind = new Point[10];
        Point[] pointsStart = new Point[10];
        double[] normSquared = new double[10];
        Mat[] Original = new Mat[10];

        #region BestBox Class
        public class BestBox
        {
            public RotatedRect Box { get; set; }
            public double Value { get; set; }
            public VectorOfPoint VectorsList { get; set; }

            public BestBox(RotatedRect box, double value, VectorOfPoint vector)
            {
                Box = box;
                Value = value;
                VectorsList = vector;
            }
        }
        #endregion

        public Main()
        {
            InitializeComponent();
        }

        private void input()
        {
            double upperThershold, lowerThreshold;
            upperThershold = double.Parse(textBox1.Text);
            lowerThreshold = double.Parse(textBox2.Text);
        }

        //------------------Main Algorithm----------------------//
        private VectorOfPoint Shape1(Mat FrameFormCam)
        {

            StringBuilder msgBuilder = new StringBuilder("Performance: "); //start clock to measure performance

            Mat imageread = FrameFormCam;
            int original_Height = imageread.Height;
            int original_Width = imageread.Width;
            CvInvoke.Resize(imageread, imageread, new Size(640, 480)); // Resize/ Interpolation of our image to reduce the complexity

            //---------Convert the image to grayscale and filter out the noise-------//
            Mat image = new Mat();
            CvInvoke.CvtColor(imageread, image, ColorConversion.Bgr2Gray);
            //-----------------------------------------------------------------------//

            //------use image pyramid to remove noise----------//
            Mat lowerImage = new Mat();
            Mat pyrDown = new Mat();

            CvInvoke.PyrDown(image, pyrDown);
            CvInvoke.PyrUp(pyrDown, lowerImage);
            //------------------------------------------------//

            //----------unsharp image to enhance contrast-------//
            Size ksize = new Size(3, 3);
            double aplha = 1.5;
            double beta = -0.5;
            double gamma = 0;
            CvInvoke.AddWeighted(image, aplha, lowerImage, beta, gamma, lowerImage);
            //-------------------------------------------------//

            //-----------Lower Image Specs-------------//
            int Height = lowerImage.Height;
            int Width = lowerImage.Width;
            //----------------------------------------//

            //----------Center of the image----------//
            double x_image = Width / 2;
            double y_image = Height / 2;

            #region Canny and edge detection
            Stopwatch watch = Stopwatch.StartNew(); //Time elapsed. It is useful for micro-benchmarks in code optimization.

            #region Read Canny Thresholds from User or take default
            double cannyThreshold, cannyThresholdLinking;
            try // read thresholds from user, if empty take default thresholds
            {
                cannyThreshold = double.Parse(textBox1.Text);
                cannyThresholdLinking = double.Parse(textBox2.Text);
            }
            catch (System.FormatException e)
            {
                cannyThreshold = 120;
                cannyThresholdLinking = 60;
            }
            #endregion

            Mat cannyEdges = new Mat();
            CvInvoke.Canny(lowerImage, cannyEdges, cannyThreshold, cannyThresholdLinking); // Canny Edge Detection

            watch.Stop();
            msgBuilder.Append(String.Format("Canny - {0} ms; ", watch.ElapsedMilliseconds));
            #endregion

            #region Find Approximate Rectangles

            watch.Reset(); watch.Start();
            List<RotatedRect> boxList = new List<RotatedRect>(); // list of the Minimum Area Rectangles of contours
            List<VectorOfPoint> ListOfBestBoxContours = new List<VectorOfPoint>(); // list of our contours
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint(); // A vector of vector of points
            Random rng = new Random();

            Mat Dilated = new Mat();
            Mat rect_6 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(5, 5), new Point(3, 3));
            CvInvoke.Dilate(cannyEdges, Dilated, rect_6, new Point(-1, -1), 5, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);  // Dilate the image 5 times
            Mat Eroded = new Mat();
            CvInvoke.Erode(Dilated, Eroded, rect_6, new Point(-1, -1), 4, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue); // Erode the image 4 times

            Mat cresult = new Mat(cannyEdges.Size, DepthType.Cv8U, 3);
            CvInvoke.FindContours(Eroded, contours, null, RetrType.List, ChainApproxMethod.ChainApproxNone); // Finds the contours of the image

            int count = contours.Size;
            for (int i = 0; i < count; i++)
            {
                using (VectorOfPoint contour = contours[i])
                using (VectorOfPoint approxContour = new VectorOfPoint())
                {
                    CvInvoke.ApproxPolyDP(contour, approxContour, 5, true);
                    if (CvInvoke.ContourArea(approxContour, true) > 6500) // only consider contours with area greater than 6500 pixels
                    {
                        if (approxContour.Size > 3) //The contour has > 3 vertices.
                        {
                            ListOfBestBoxContours.Add(contour); // add candidate contour to the list
                            boxList.Add(CvInvoke.MinAreaRect(approxContour)); // add the Minimum Area Rectangle of the candidate contour to the list
                        }
                    }
                }
            }
            #endregion

            try
            {
                #region find best rectangle and take its contour

                #region find the minArea,MaxArea,minDistance,maxDistance for the Rescaling

                double maxArea = 0; // maximum area amongst the rectangles representing the contours
                double minArea = 10000000000; // minimum area amongst the rectangles representing the contours
                double maxDistance = 0; // the distance of the rectangle witch is further away from the center of the frame from it
                double minDistance = 10000000000; // the distance of the rectangle witch is close to the center of the frame from it
                List<double> value = new List<double>();

                //find the minArea,MaxArea,minDistance,maxDistance for the Rescaling
                for (int z = 0; z < boxList.Count; z++)
                {
                    maxArea = Math.Max(maxArea, ((boxList[z].Size.Height) * (boxList[z].Size.Width))); //maxArea = max(maxArea,area[z])
                    minArea = Math.Min(minArea, ((boxList[z].Size.Height) * (boxList[z].Size.Width))); //minArea = min(minArea,area[z])
                    maxDistance = Math.Max(maxDistance, (Math.Pow(Math.Abs((boxList[z].Center.X - x_image)), 2) + Math.Pow(Math.Abs((boxList[z].Center.Y - y_image)), 2))); //maxDistance = max(maxDistance,DistanceFromTheCenterOfTheImage[z])
                    minDistance = Math.Min(minDistance, (Math.Pow(Math.Abs((boxList[z].Center.X - x_image)), 2) + Math.Pow(Math.Abs((boxList[z].Center.Y - y_image)), 2))); //minDistance = min(minDistance,DistanceFromTheCenterOfTheImage[z])
                }
                #endregion

                #region Find the Values of each Rectangle
                foreach (RotatedRect y in boxList)
                    // Rescaled(Area[z]) = (Area[z] - MinArea )/(MaxArea-MinArea)  ,  Rescaled(Area[z]) e [0,1]
                    // Rescaled(DistanceFromCenterOfImage[z]) = (DistanceFromCenterOfImage[z] - minDistance )/(maxDistance-minDistance)  ,  Rescaled(DistanceFromCenterOfImage[z]) e [0,1]
                    // List of value = 0.5 * Rescaled(Area[z]) + 0.5 * (1-Rescaled(DistanceFromCenterOfImage[z]))
                    value.Add(((((y.Size.Height) * (y.Size.Width)) - minArea) / (maxArea - minArea)) * 0.5 + 0.5 * (1 - (((Math.Pow(Math.Abs(y.Center.X - x_image), 2) + Math.Pow(Math.Abs(y.Center.Y - y_image), 2)) - minDistance) / (maxDistance - minDistance))));
                #endregion

                #region Insert in class BestBox and sort it by descending value
                List<BestBox> Best = new List<BestBox>(); //A Structure for finding the best contour. Deffinition on line 32
                for (int i = 0; i < value.Count; i++)
                {
                	// add the rectangle, its contour and its value
                    Best.Add(new BestBox(boxList[i], value[i], ListOfBestBoxContours[i]));
                }

                Best.Sort(delegate (BestBox x, BestBox y)
                {
                    //Sort Descending by value
                    return y.Value.CompareTo(x.Value);
                });
                #endregion

                #endregion

                Image<Bgr, Byte> Filling1 = lowerImage.ToImage<Bgr, Byte>().CopyBlank(); // a blank image to draw on and fill the best contour
                CvInvoke.FillConvexPoly(Filling1, Best[0].VectorsList, new MCvScalar(0, 0, 255)); // Fill the inside of the Best Contour to make it solid

                CvInvoke.Canny(Filling1, cannyEdges, cannyThreshold, cannyThresholdLinking); // Canny Edge Detection

                //clear the lists and structs
                boxList.Clear();
                ListOfBestBoxContours.Clear();
                Best.Clear();
                value.Clear();

                #region Lough Lines

                #region Compute Hough Lines on best contour region
                //------------------Compute HoughLines----------------//
                LineSegment2D[] lines = CvInvoke.HoughLinesP(
                cannyEdges,
                1, //Distance resolution in pixel-related units
                Math.PI / 180, //Angle resolution measured in radians.
                40, //threshold
                0, //min Line width
                15); //gap between lines
                //---------------------------------------------------//
                #endregion

                #region Extend the Hough Lines
                Point A = new Point();
                Point B = new Point();
                Point CB = new Point();
                Point CA = new Point();
                int length = 500;

                // extending the hough lines, allowing them to intesect with each other
                for (int i = 0; i < lines.Length; i++)
                {
                    A = lines[i].P1;
                    B = lines[i].P2;
                    CB.X = (int)(B.X + (B.X - A.X) / lines[i].Length * length);
                    CB.Y = (int)(B.Y + (B.Y - A.Y) / lines[i].Length * length);
                    CA.X = (int)(A.X + (A.X - B.X) / lines[i].Length * length);
                    CA.Y = (int)(A.Y + (A.Y - B.Y) / lines[i].Length * length);
                    lines[i].P1 = CA;
                    lines[i].P2 = CB;
                }
                #endregion

                #region Draw Hough Lines representing our area on a blank image
                Image<Bgr, Byte> DrawLines3 = lowerImage.ToImage<Bgr, Byte>().CopyBlank();
                foreach (LineSegment2D line in lines)
                    DrawLines3.Draw(line, new Bgr(Color.White), 2); // Draw these lines on a blank image
                CvInvoke.CvtColor(DrawLines3, DrawLines3, ColorConversion.Bgr2Gray);
                #endregion

                #endregion

                #region Find new Contours and Rectangle areas

                VectorOfVectorOfPoint contours1 = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(DrawLines3.Mat, contours1, null, RetrType.List, ChainApproxMethod.ChainApproxNone); //find the contours
                VectorOfPoint contour;
                count = contours1.Size;
                for (int i = 0; i < count; i++)
                {
                    contour = contours1[i];
                    using (VectorOfPoint approxContour = new VectorOfPoint())
                    {
                        CvInvoke.ApproxPolyDP(contour, approxContour, 5, true);
                        if (CvInvoke.ContourArea(approxContour, true) > 6500) //only consider contours with area greater than 6500
                        {
                            if (approxContour.Size > 3) //The contour has > 3 vertices.
                            {
                                ListOfBestBoxContours.Add(contour); // add candidate contour to the list
                                boxList.Add(CvInvoke.MinAreaRect(approxContour)); // add the Minimum Area Rectangle of the candidate contour to the list

                            }
                        }
                    }
                }
                #endregion

                #region find best rectangle and take its contour

                maxArea = 0;
                minArea = 10000000000;
                maxDistance = 0;
                minDistance = 10000000000;

                #region find the minArea,MaxArea,minDistance,maxDistance for the Rescaling
                for (int z = 0; z < boxList.Count; z++)
                {
                    maxArea = Math.Max(maxArea, ((boxList[z].Size.Height) * (boxList[z].Size.Width))); //maxArea = max(maxArea,area[z])
                    minArea = Math.Min(minArea, ((boxList[z].Size.Height) * (boxList[z].Size.Width))); //minArea = min(minArea,area[z])
                    maxDistance = Math.Max(maxDistance, (Math.Pow(Math.Abs((boxList[z].Center.X - x_image)), 2) + Math.Pow(Math.Abs((boxList[z].Center.Y - y_image)), 2))); //maxDistance = max(maxDistance,DistanceFromTheCenterOfTheImage[z])
                    minDistance = Math.Min(minDistance, (Math.Pow(Math.Abs((boxList[z].Center.X - x_image)), 2) + Math.Pow(Math.Abs((boxList[z].Center.Y - y_image)), 2))); //minDistance = min(minDistance,DistanceFromTheCenterOfTheImage[z])
                }
                #endregion

                #region Find the Values of each Rectangle
                foreach (RotatedRect y in boxList)
                    // Rescaled(Area[z]) = (Area[z] - MinArea )/(MaxArea-MinArea)  ,  Rescaled(Area[z]) e [0,1]
                    // Rescaled(DistanceFromCenterOfImage[z]) = (DistanceFromCenterOfImage[z] - minDistance )/(maxDistance-minDistance)  ,  Rescaled(DistanceFromCenterOfImage[z]) e [0,1]
                    // List of value = 0.5 * Rescaled(Area[z]) + 0.5 * (1-Rescaled(DistanceFromCenterOfImage[z]))
                    value.Add(((((y.Size.Height) * (y.Size.Width)) - minArea) / (maxArea - minArea)) * 0.5 + 0.5 * (1 - (((Math.Pow(Math.Abs(y.Center.X - x_image), 2) + Math.Pow(Math.Abs(y.Center.Y - y_image), 2)) - minDistance) / (maxDistance - minDistance))));
                #endregion

                #region Insert in class BestBox and sort it by descending value
                for (int i = 0; i < value.Count; i++)
                {
                	// add the rectangle, its contour and its value
                    Best.Add(new BestBox(boxList[i], value[i], ListOfBestBoxContours[i]));
                }

                Best.Sort(delegate (BestBox x, BestBox y)
                {
                    //Sort Descending by value
                    return y.Value.CompareTo(x.Value);
                });
                #endregion

                #endregion

                contour = Best[0].VectorsList; //the first element is always the best contour region. This is the final desired area in the frame
                return contour;

            }
            catch (Exception e) when (e is System.InvalidOperationException || e is System.ArgumentOutOfRangeException)
            {
                Mat result = imageread;
                return null;
            }


        }

        //searcher with points as input
        private Point[] searcher(Point[] input_points, Mat input_image1, Mat input_image2, ref Mat[] templ)
        {

            Mat image1 = input_image1.Clone();
            Mat image2 = input_image2.Clone();

            //Convert the image to grayscale and filter out the noise
            CvInvoke.CvtColor(image1, image1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(image2, image2, ColorConversion.Bgr2Gray);

            //blur
            Mat imageread = new Mat();
            Mat pyrDown1 = new Mat();
            Mat imageread2 = new Mat();
            Mat pyrDown2 = new Mat();

            CvInvoke.PyrDown(image1, pyrDown1);
            CvInvoke.PyrUp(pyrDown1, imageread);
            CvInvoke.PyrDown(image2, pyrDown2);
            CvInvoke.PyrUp(pyrDown2, imageread2);

            //unsharp
            //Size ksize = new Size(3, 3);
            double aplha = 1.5;
            double beta = -0.5;
            double gamma = 0;
            CvInvoke.AddWeighted(image1, aplha, imageread, beta, gamma, imageread);
            CvInvoke.AddWeighted(image2, aplha, imageread2, beta, gamma, imageread2);

            //
            int patternWindow = 30;//the template size
            int searchWindow = 150;//the search window size
            Mat[] recs = templ;//where the templates are stored
            Mat[] img = new Mat[input_points.Length];//where the search windows are stored
            Point[] points = new Point[input_points.Length];//where the new edges are stored
            for (int i = 0; i < input_points.Length; i++)//for each corner in the starting contour, search for the most similar point in the next frame
            {
                img[i] = new Mat();
                CvInvoke.GetRectSubPix(imageread2, new System.Drawing.Size(searchWindow, searchWindow), input_points[i], img[i]);//cut the search windows from image 2

                Mat outp = new Mat();//the match template output
                CvInvoke.MatchTemplate(img[i], recs[i], outp, TemplateMatchingType.CcoeffNormed);// match the template inside the window
                //CvInvoke.MatchTemplate(img[i], recs[i], outp, TemplateMatchingType.SqdiffNormed);// match the template inside the window

                double minVal = 0; double maxVal = 0; Point minLoc = new Point();
                CvInvoke.MinMaxLoc(outp, ref minVal, ref maxVal, ref minLoc, ref points[i]);//find the locations of min and max similarity and their values
                //CvInvoke.MinMaxLoc(outp, ref minVal, ref maxVal, ref points[i], ref minLoc);//find the locations of min and max similarity and their values

                int printpointX = input_points[i].X + points[i].X - searchWindow / 2 + patternWindow / 2;//translation from X coord of search window to X coord of image2
                int printpointY = input_points[i].Y + points[i].Y - searchWindow / 2 + patternWindow / 2;//translation from Y coord of search window to Y coord of image2
                points[i] = new Point(printpointX, printpointY);//the final point
                CvInvoke.Circle(input_image2, points[i], (int)(patternWindow / 3), new MCvScalar(132, 255, 122));//print a circle around the search point
            }

            //Draw Lines
            Image<Bgr, Byte> Draw = input_image2.ToImage<Bgr, Byte>();
            if (points.Length > 0)
            {
                int i;
                for (i = 1; i < points.Length; i++)
                {
                    Draw.Draw(new LineSegment2D(points[i - 1], points[i]), new Bgr(Color.White), 1);
                }
                Draw.Draw(new LineSegment2D(points[i - 1], points[0]), new Bgr(Color.White), 1);
            }

            imageBox2.Image = Draw.Mat;//the output image with circles printed around new corners
            return points;
        }

        //searcher with contour as input
        private Point[] searcher(VectorOfPoint contour, Mat input_image1, Mat input_image2, ref Mat[] templ)
        {
            Mat image1 = input_image1.Clone();
            Mat image2 = input_image2.Clone();

            //Convert the image to grayscale and filter out the noise
            CvInvoke.CvtColor(image1, image1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(image2, image2, ColorConversion.Bgr2Gray);

            //blur
            Mat imageread = new Mat();
            Mat pyrDown1 = new Mat();
            Mat imageread2 = new Mat();
            Mat pyrDown2 = new Mat();

            CvInvoke.PyrDown(image1, pyrDown1);
            CvInvoke.PyrUp(pyrDown1, imageread);
            CvInvoke.PyrDown(image2, pyrDown2);
            CvInvoke.PyrUp(pyrDown2, imageread2);

            //unsharp
            Size ksize = new Size(3, 3);
            double aplha = 1.5;
            double beta = -0.5;
            double gamma = 0;
            CvInvoke.AddWeighted(image1, aplha, imageread, beta, gamma, imageread);
            CvInvoke.AddWeighted(image2, aplha, imageread2, beta, gamma, imageread2);

            //find corners

            VectorOfPoint contour2 = new VectorOfPoint();
            CvInvoke.ApproxPolyDP(contour, contour2, CvInvoke.ArcLength(contour, true) * 0.05, true);//approximate the contour to each main lines
            Point[] pts = contour2.ToArray();//make it array from vector
            LineSegment2D[] edges = PointCollection.PolyLine(pts, true);//create array of LineSegment2D for each of the main lines
            Point[] gwnies = new Point[edges.Length];//where the edges of the contour are stored
            for (int i = 0; i < gwnies.Length; i++)//for every 2 consecutive lines calculate the intersection point with gwnia(LineSegment2D, LineSegment2D) function
            {
                if (gwnies.Length == i + 1)
                {
                    gwnies[i] = gwnia(edges[i], edges[0]);
                }
                else
                {
                    gwnies[i] = gwnia(edges[i], edges[i + 1]);
                }
                if (gwnies[i].X < 0)
                {
                    return null;
                }
            }

            //
            int patternWindow = 30;//the template size
            int searchWindow = 150;//the search window size
            Mat[] recs = new Mat[edges.Length];//where the templates are stored
            Mat[] img = new Mat[edges.Length];//where the search windows are stored
            Point[] points = new Point[edges.Length];//where the new edges are stored
            for (int i = 0; i < gwnies.Length; i++)//for each corner in the starting contour, search for the most similar point in the next frame
            {
                recs[i] = new Mat();
                img[i] = new Mat();
                CvInvoke.GetRectSubPix(imageread, new System.Drawing.Size(patternWindow, patternWindow), gwnies[i], recs[i]);//cut the template from image 1
                CvInvoke.GetRectSubPix(imageread2, new System.Drawing.Size(searchWindow, searchWindow), gwnies[i], img[i]);//cut the search windows from image 2

                Mat outp = new Mat();//the match template output
                CvInvoke.MatchTemplate(img[i], recs[i], outp, TemplateMatchingType.CcoeffNormed);// match the template inside the window
                //CvInvoke.MatchTemplate(img[i], recs[i], outp, TemplateMatchingType.SqdiffNormed);// match the template inside the window

                double minVal = 0; double maxVal = 0; Point minLoc = new Point();
                CvInvoke.MinMaxLoc(outp, ref minVal, ref maxVal, ref minLoc, ref points[i]);//find the locations of min and max similarity and their values
                //CvInvoke.MinMaxLoc(outp, ref minVal, ref maxVal, ref points[i], ref minLoc);//find the locations of min and max similarity and their values
                int printpointX = gwnies[i].X + points[i].X - searchWindow / 2 + patternWindow / 2;//translation from X coord of search window to X coord of image2
                int printpointY = gwnies[i].Y + points[i].Y - searchWindow / 2 + patternWindow / 2;//translation from Y coord of search window to Y coord of image2
                points[i] = new Point(printpointX, printpointY);//the final point
                CvInvoke.Circle(input_image2, points[i], (int)(patternWindow / 3), new MCvScalar(132, 255, 122));//print a circle around the search point ////////////////////
            }
            templ = recs;

            //Draw Lines
            Image<Bgr, Byte> Draw = input_image2.ToImage<Bgr, Byte>(); //////////
            if (points.Length > 0)
            {
                int i;
                for (i = 1; i < points.Length; i++)
                {
                    Draw.Draw(new LineSegment2D(points[i - 1], points[i]), new Bgr(Color.White), 1);
                }
                Draw.Draw(new LineSegment2D(points[i - 1], points[0]), new Bgr(Color.White), 1);
            }

            imageBox2.Image = Draw.Mat;//the output image with circles printed around new corners
            return points;
        }

        //given 2 lines in LineSegment2d form, find the intersection Point and return it
        //throws exception if lines are parallel
        private Point gwnia(LineSegment2D edges0, LineSegment2D edges1)
        {
            float A0 = edges0.P1.Y - edges0.P2.Y;
            float B0 = edges0.P2.X - edges0.P1.X;
            float C0 = A0 * edges0.P2.X + B0 * edges0.P2.Y;
            float A1 = edges1.P1.Y - edges1.P2.Y;
            float B1 = edges1.P2.X - edges1.P1.X;
            float C1 = A1 * edges1.P2.X + B1 * edges1.P2.Y;

            float delta = A0 * B1 - A1 * B0;
            if (delta == 0)
            {
                Console.WriteLine("Lines are parallel");
                return new Point(-1, -1);
            }

            // now return the Vector2 intersection point
            return new Point((int)((B1 * C0 - B0 * C1) / delta), (int)((A0 * C1 - A1 * C0) / delta));
        }

        private void fontDialog1_Apply(object sender, EventArgs e)
        {

        }

        private void Main_Load(object sender, EventArgs e)
        {

        }

        private void textBox1_KeyPress(object sender, KeyPressEventArgs e)
        {
            char ch = e.KeyChar;

            if (ch == 46 && textBox1.Text.IndexOf('.') != -1)
            {
                e.Handled = true;
                return;
            }

            if (!Char.IsDigit(ch) && ch != 8 && ch != 46)
            {
                e.Handled = true;
            }
        }

        private void textBox2_KeyPress(object sender, KeyPressEventArgs e)
        {
            char ch = e.KeyChar;

            if (ch == 46 && textBox1.Text.IndexOf('.') != -1)
            {
                e.Handled = true;
                return;
            }

            if (!Char.IsDigit(ch) && ch != 8 && ch != 46)
            {
                e.Handled = true;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {

            #region if capture is not created, create it now
            if (capture == null)
            {
                try
                {
                    capture = new Capture(0);
                    capture.ImageGrabbed += ProcessFrame;
                }
                catch (NullReferenceException excpt)
                {
                    MessageBox.Show(excpt.Message);
                }
            }
            #endregion

            if (capture != null)
            {
                if (captureInProgress)
                {  //if camera is getting frames then stop the capture and set button Text
                    // "Start" for resuming capture
                    State = 0;
                    capture.Pause();
                }
                else
                {
                    //if camera is NOT getting frames then start the capture and set button
                    // Text to "Stop" for pausing capture
                    capture.Start();
                }

                captureInProgress = !captureInProgress;
            }
        }

        //---------------------ExampleMethod-----------------------------------------------//
        //Process Frame() below is our user defined function in which we will create an EmguCv 
        //type image called ImageFrame. capture a frame from camera and allocate it to our 
        //ImageFrame. then show this image in ourEmguCV imageBox
        //------------------------------------------------------------------------------//
        private void ProcessFrame(object sender, EventArgs arg)
        {
            Mat frame = new Mat();
            capture.Retrieve(frame, 0);
            imageBox1.Image = frame;

            if (State == 0)
            {
                imageBox2.Image = frame;
                contours = Shape1(frame); // run the Shape function to find the area
                if (contours != null)
                {
                    PreviousFrame = frame.Clone();
                    State++;
                }
            }
            else if (State == 1)
            {
                points = searcher(contours, PreviousFrame, frame, ref templ); // template matching to the second frame after the area is found 
                pointsStart = points;
                try
                {
                	//------------Calculate the norm of each of the corner teplates of the second frame-------------//
                    for (int i = 0; i < points.Length; i++)
                    {
                        Original[i] = new Mat();
                        CvInvoke.GetRectSubPix(frame, new System.Drawing.Size(20, 20), points[i], Original[i]);
                        normSquared[i] = Original[i].Dot(Original[i]);
                    }
                    //---------------------------------------------------------------------------------------------//
                }
                catch (System.NullReferenceException e)
                {
                    Console.WriteLine("NullReferenceException in points. Line 669");
                    points = null;
                }

                PreviousFrame = frame.Clone();
                if (points != null)
                {
                    State++;
                }
                else
                {
                    State = 0;
                }
            }
            else
            {
                points = searcher(points, PreviousFrame, frame, ref templ); // template matching to each next frame 
                PreviousFrame = frame.Clone();

                if (framecount5 == 0)
                {
                    points5behind = points;
                }
                else if (framecount5 == 1)
                {
                    for (int i = 0; i < points.Length; i++)
                    {

                        Mat CurrentFrame = new Mat();
                        CvInvoke.GetRectSubPix(frame, new System.Drawing.Size(20, 20), points[i], CurrentFrame);
                        double ratio = CurrentFrame.Dot(Original[i]) / normSquared[i]; // ratio betwwen the norm of the first template and the Dot product between the first and new template
                        if (ratio < 0.9 || ratio > 1.1) //error checking
                        {
                            State = 0;
                            Console.WriteLine(ratio);
                            Console.WriteLine("Refresh cause Norm");
                            break;
                        }


                        if (points.Length == i + 1)
                        {
                        	//calculate distances
                            double tmp3 = Math.Sqrt(Math.Pow(points[0].X - points[i].X, 2) + Math.Pow(points[0].Y - points[i].Y, 2));
                            double tmp1 = Math.Sqrt(Math.Pow(pointsStart[0].X - pointsStart[i].X, 2) + Math.Pow(pointsStart[0].Y - pointsStart[i].Y, 2));
                            double tmp2 = Math.Sqrt(Math.Pow(points5behind[0].X - points5behind[i].X, 2) + Math.Pow(points5behind[0].Y - points5behind[i].Y, 2));
                            ratio = tmp1 / tmp2;
                            double ratio2 = tmp3 / tmp2;
                            if ((ratio2 < 0.9 || ratio2 > 1.1) || (ratio < 0.9 || ratio > 1.1)) //error checking
                            {
                                State = 0;
                                Console.WriteLine(ratio);
                                Console.WriteLine("Refresh");
                                break;
                            }
                        }
                        else
                        {
                        	//calculate distances
                            double tmp3 = Math.Sqrt(Math.Pow(points[i + 1].X - points[i].X, 2) + Math.Pow(points[i + 1].Y - points[i].Y, 2));
                            double tmp1 = Math.Sqrt(Math.Pow(pointsStart[i + 1].X - pointsStart[i].X, 2) + Math.Pow(pointsStart[i + 1].Y - pointsStart[i].Y, 2));
                            double tmp2 = Math.Sqrt(Math.Pow(points5behind[i + 1].X - points5behind[i].X, 2) + Math.Pow(points5behind[i + 1].Y - points5behind[i].Y, 2));
                            ratio = tmp1 / tmp2;
                            double ratio2 = tmp3 / tmp2;
                            if ((ratio2 < 0.9 || ratio2 > 1.1) || (ratio < 0.9 || ratio > 1.1)) //error checking
                            {
                                State = 0;
                                Console.WriteLine(ratio);
                                Console.WriteLine("Refresh");
                                break;
                            }
                        }

                    }
                    framecount5 = -1;
                }
                framecount5++;
            }
            frame.Dispose();
        }

        private void ReleaseData()
        {
            if (capture != null)
                capture.Dispose();
        }

    }
}