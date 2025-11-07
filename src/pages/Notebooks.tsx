import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import NotebookCard from "@/components/NotebookCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileCode, Search, TrendingUp, Award, Plus } from "lucide-react";
import { useState, useRef } from "react";
import { Link } from "react-router-dom";

const Notebooks = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const tutorialsRef = useRef<HTMLDivElement>(null);

  const notebooks = [
    {
      id: "1",
      title: "Exploratory Data Analysis - Ghana Agriculture",
      author: "Kwame Mensah",
      description: "Comprehensive EDA of Ghana's agricultural data with visualizations and statistical insights.",
      likes: 234,
      comments: 45,
      views: 3421,
      tags: ["EDA", "Agriculture", "Visualization"],
      language: "Python",
      votes: 89,
      featured: true,
    },
    {
      id: "2",
      title: "XGBoost Model for Crop Yield Prediction",
      author: "Ama Osei",
      description: "Step-by-step implementation of XGBoost with hyperparameter tuning for crop yield forecasting.",
      likes: 189,
      comments: 32,
      views: 2156,
      tags: ["Machine Learning", "XGBoost", "Tutorial"],
      language: "Python",
      votes: 67,
      featured: true,
    },
    {
      id: "3",
      title: "Feature Engineering Techniques",
      author: "Kofi Addo",
      description: "Advanced feature engineering methods for improving model performance on agricultural datasets.",
      likes: 156,
      comments: 28,
      views: 1876,
      tags: ["Feature Engineering", "Best Practices"],
      language: "Python",
      votes: 54,
      featured: false,
    },
    {
      id: "4",
      title: "Time Series Analysis with LSTM",
      author: "Abena Frimpong",
      description: "Deep learning approach using LSTM networks for temporal patterns in climate data.",
      likes: 201,
      comments: 41,
      views: 2543,
      tags: ["Deep Learning", "LSTM", "Time Series"],
      language: "Python",
      votes: 72,
      featured: false,
    },
    {
      id: "5",
      title: "Ensemble Methods Comparison",
      author: "Yaw Boateng",
      description: "Comparing Random Forest, Gradient Boosting, and Stacking techniques for yield prediction.",
      likes: 143,
      comments: 24,
      views: 1654,
      tags: ["Ensemble", "Comparison", "Model Selection"],
      language: "R",
      votes: 48,
      featured: false,
    },
    {
      id: "6",
      title: "Data Cleaning Pipeline",
      author: "Efua Asare",
      description: "Robust data cleaning and preprocessing pipeline for handling missing values and outliers.",
      likes: 167,
      comments: 19,
      views: 1432,
      tags: ["Data Cleaning", "Preprocessing", "Pipeline"],
      language: "Python",
      votes: 61,
      featured: false,
    },
  ];

  const filteredNotebooks = notebooks.filter(notebook =>
    notebook.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    notebook.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    notebook.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <div className="flex-1">
        <section className="bg-gradient-to-br from-primary/10 via-secondary/5 to-accent/10 border-b border-border py-16">
          <div className="container mx-auto px-4">
            <div className="max-w-4xl mx-auto text-center">
              <div className="flex justify-center mb-6">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary">
                  <FileCode className="h-10 w-10 text-primary-foreground" />
                </div>
              </div>
              <h1 className="text-5xl font-bold mb-4">Code Notebooks</h1>
              <p className="text-xl text-muted-foreground mb-8">
                Explore, learn, and share data science notebooks and analysis with the community
              </p>
              <div className="flex gap-4 justify-center">
                <Link to="/notebooks/create">
                  <Button size="lg">
                    <Plus className="mr-2 h-4 w-4" />
                    Create New Notebook
                  </Button>
                </Link>
                <Button 
                  size="lg" 
                  variant="outline"
                  onClick={() => {
                    tutorialsRef.current?.scrollIntoView({ behavior: 'smooth' });
                    setSearchQuery("Tutorial");
                  }}
                >
                  Browse Tutorials
                </Button>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12" ref={tutorialsRef}>
          <div className="container mx-auto px-4">
            <div className="max-w-6xl mx-auto space-y-6">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search notebooks, authors, or tags..."
                  className="pl-10"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              <Tabs defaultValue="popular" className="w-full">
                <TabsList>
                  <TabsTrigger value="popular">
                    <TrendingUp className="h-4 w-4 mr-2" />
                    Most Popular
                  </TabsTrigger>
                  <TabsTrigger value="featured">
                    <Award className="h-4 w-4 mr-2" />
                    Featured
                  </TabsTrigger>
                  <TabsTrigger value="recent">Recent</TabsTrigger>
                  <TabsTrigger value="votes">Most Voted</TabsTrigger>
                </TabsList>

                <TabsContent value="popular" className="mt-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {filteredNotebooks.map((notebook) => (
                      <NotebookCard key={notebook.id} notebook={notebook} />
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="featured" className="mt-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {filteredNotebooks.filter(n => n.featured).map((notebook) => (
                      <NotebookCard key={notebook.id} notebook={notebook} showBorder />
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="recent" className="mt-6">
                  <p className="text-center text-muted-foreground py-8">Recent notebooks will appear here</p>
                </TabsContent>

                <TabsContent value="votes" className="mt-6">
                  <p className="text-center text-muted-foreground py-8">Most voted notebooks will appear here</p>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
};

export default Notebooks;
