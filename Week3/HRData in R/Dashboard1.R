library(shiny)
library(ggplot2)
library(shinydashboard)


ui <- fluidPage(
  titlePanel("HR Management Analytics Dashboard"),
  sidebarLayout(
    sidebarPanel(
      selectInput("department", "Select Department:", 
                  choices = unique(data$Department))
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Overview", 
                 fluidRow(
                   valueBoxOutput("total_employees"),
                   valueBoxOutput("avg_salary"),
                   valueBoxOutput("avg_age")
                 ),
                 plotOutput("gender_distribution"),
                 plotOutput("age_distribution"),
                 plotOutput("education_distribution"),
                 plotOutput("attrition_by_dept"),
                 plotOutput("job_satisfaction"),
                 plotOutput("performance_ratings"),
                 plotOutput("training_times")
        ),
        tabPanel("Employee Engagement", 
                 plotOutput("engagement_by_dept"),
                 plotOutput("years_with_manager"),
                 plotOutput("work_life_balance")
        ),
        tabPanel("Attrition Analysis", 
                 plotOutput("attrition_by_age"),
                 plotOutput("attrition_by_job_role"),
                 plotOutput("attrition_by_gender")
        )
      )
    )
  )
)

server <- function(input, output) {
  filtered_data <- reactive({
    data[data$Department == input$department, ]
  })
  
  output$total_employees <- renderValueBox({
    valueBox(
      nrow(filtered_data()), "Total Employees", icon = icon("users"),
      color = "blue"
    )
  })
  
  output$avg_salary <- renderValueBox({
    avg_salary <- round(mean(filtered_data()$MonthlyIncome), 2)
    valueBox(
      avg_salary, "Average Salary", icon = icon("dollar-sign"),
      color = "green"
    )
  })
  
  output$avg_age <- renderValueBox({
    avg_age <- round(mean(filtered_data()$Age), 2)
    valueBox(
      avg_age, "Average Age", icon = icon("calendar-alt"),
      color = "purple"
    )
  })
  
  output$gender_distribution <- renderPlot({
    gender_dist <- table(filtered_data()$Gender)
    pie(gender_dist, main="Gender Distribution", 
        col=c("pink", "lightblue"), labels=paste0(names(gender_dist), "\n", 
                                                  round(prop.table(gender_dist)*100, 2), "%"))
  })
  
  output$age_distribution <- renderPlot({
    ggplot(filtered_data(), aes(x=Age)) + 
      geom_histogram(binwidth=5, fill="blue", color="black") +
      labs(title="Age Distribution", x="Age", y="Count")
  })
  
  output$education_distribution <- renderPlot({
    ggplot(filtered_data(), aes(x=EducationField)) + 
      geom_bar(fill="orange", color="black") +
      labs(title="Education Distribution", x="Education Field", y="Count")
  })
  
  output$attrition_by_dept <- renderPlot({
    ggplot(filtered_data(), aes(x=Department, fill=Attrition)) + 
      geom_bar(position="fill") +
      labs(title="Attrition by Department", x="Department", y="Proportion")
  })
  
  output$job_satisfaction <- renderPlot({
    ggplot(filtered_data(), aes(x=JobSatisfaction)) + 
      geom_bar(fill="green", color="black") +
      labs(title="Job Satisfaction Levels", x="Job Satisfaction", y="Count")
  })
  
  output$performance_ratings <- renderPlot({
    ggplot(filtered_data(), aes(x=PerformanceRating)) + 
      geom_bar(fill="purple", color="black") +
      labs(title="Performance Ratings Distribution", x="Performance Rating", y="Count")
  })
  
  output$training_times <- renderPlot({
    ggplot(filtered_data(), aes(x=TrainingTimesLastYear)) + 
      geom_bar(fill="orange", color="black") +
      labs(title="Training Times Last Year", x="Training Times", y="Count")
  })
  
  output$engagement_by_dept <- renderPlot({
    ggplot(filtered_data(), aes(x=Department, y=JobInvolvement)) + 
      geom_boxplot(fill="lightblue", color="black") +
      labs(title="Employee Engagement by Department", x="Department", y="Job Involvement")
  })
  
  output$years_with_manager <- renderPlot({
    ggplot(filtered_data(), aes(x=YearsWithCurrManager)) + 
      geom_histogram(binwidth=1, fill="lightgreen", color="black") +
      labs(title="Years with Current Manager", x="Years", y="Count")
  })
  
  output$work_life_balance <- renderPlot({
    ggplot(filtered_data(), aes(x=WorkLifeBalance)) + 
      geom_bar(fill="lightcoral", color="black") +
      labs(title="Work-Life Balance", x="Work-Life Balance", y="Count")
  })
  
  output$attrition_by_age <- renderPlot({
    ggplot(filtered_data(), aes(x=Age, fill=Attrition)) + 
      geom_histogram(binwidth=5, position="dodge") +
      labs(title="Attrition by Age", x="Age", y="Count")
  })
  
  output$attrition_by_job_role <- renderPlot({
    ggplot(filtered_data(), aes(x=JobRole, fill=Attrition)) + 
      geom_bar(position="fill") +
      labs(title="Attrition by Job Role", x="Job Role", y="Proportion")
  })
  
  output$attrition_by_gender <- renderPlot({
    ggplot(filtered_data(), aes(x=Gender, fill=Attrition)) + 
      geom_bar(position="fill") +
      labs(title="Attrition by Gender", x="Gender", y="Proportion")
  })
}

shinyApp(ui, server)